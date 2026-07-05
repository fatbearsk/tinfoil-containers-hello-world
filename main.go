// Command secretgen is a multi-tenant, high-entropy secret generator intended
// to run INSIDE a confidential computing enclave (AWS Nitro Enclaves, AMD
// SEV-SNP, Intel TDX, etc.). It replaces a demo "echo env vars" server with a
// service that mints cryptographically strong secrets on demand.
//
// What this service guarantees
//
//   - Secrets come from the OS CSPRNG via crypto/rand (getrandom(2) on Linux).
//     This is the load-bearing decision: math/rand is never used for anything
//     secret, anywhere.
//   - It FAILS CLOSED. If randomness cannot be obtained, it refuses to issue a
//     secret instead of emitting a weak one.
//   - Secrets are returned only in the HTTP response body to the caller that
//     requested them. They are never logged, never placed in a URL, and every
//     response is marked no-store. Each secret gets a non-secret ID so
//     issuance can be audited without recording the secret itself.
//
// What YOU must still supply
//
//   - Real tenant authentication. authenticateTenant is a placeholder that
//     verifies an HMAC over the tenant header. Replace it with mTLS client
//     certs or an in-enclave-validated JWT, or any caller can label itself as
//     any tenant.
//   - Remote attestation. Clients should verify the enclave BEFORE trusting it
//     with secret generation. /v1/attestation is an honest 501 stub; wire it to
//     your platform (Nitro NSM doc, SEV-SNP report, TDX quote).
//   - A seeded kernel RNG. crypto/rand is only as good as the guest kernel's
//     entropy pool; make sure the enclave runtime seeds it (virtio-rng / NSM /
//     RDSEED). Optionally set EXTRA_ENTROPY_PATH to mix a hardware source in.
//
// Quick start
//
//	go run .    # listens on :8080
//	KEY=$(head -c32 /dev/urandom | xxd -p -c256)
//	# server:
//	TENANT_AUTH_KEY=$KEY go run .
//	# client:
//	TID=acme
//	AUTH=$(printf '%s' "$TID" | openssl dgst -sha256 -mac HMAC -macopt "hexkey:$KEY" | awk '{print $2}')
//	curl -s -X POST 'http://localhost:8080/v1/secret?bits=256&format=base64url' \
//	     -H "X-Tenant-ID: $TID" -H "X-Tenant-Auth: $AUTH"
package main

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base32"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ---------------------------------------------------------------------------
// Configuration (all via environment; nothing secret is baked into the image).
// ---------------------------------------------------------------------------

type config struct {
	addr           string
	maxEntropyBits int
	defaultBits    int

	// extraEntropyPath, if set, is a hardware/enclave RNG device (e.g.
	// /dev/hwrng backed by virtio-rng, or an RDSEED-fed device exposed by the
	// runtime). Its output is combined with crypto/rand through HKDF as a
	// randomness extractor. Empty = disabled; crypto/rand alone is a sound
	// baseline, so this is strictly defense in depth.
	extraEntropyPath string

	// tenantAuthKey authenticates the tenant header in the placeholder auth
	// path. In production this should be provisioned to the enclave AFTER
	// attestation, never embedded in the build.
	tenantAuthKey []byte

	perTenantRPS   float64
	perTenantBurst int
}

func loadConfig() config {
	c := config{
		addr:             envOr("LISTEN_ADDR", ":8080"),
		maxEntropyBits:   envInt("MAX_ENTROPY_BITS", 4096),
		defaultBits:      envInt("DEFAULT_ENTROPY_BITS", 256),
		extraEntropyPath: os.Getenv("EXTRA_ENTROPY_PATH"),
		perTenantRPS:     envFloat("PER_TENANT_RPS", 5),
		perTenantBurst:   envInt("PER_TENANT_BURST", 10),
	}
	if k := os.Getenv("TENANT_AUTH_KEY"); k != "" {
		// Prefer a hex-encoded 16+ byte key; fall back to raw bytes.
		if decoded, err := hex.DecodeString(k); err == nil && len(decoded) >= 16 {
			c.tenantAuthKey = decoded
		} else {
			c.tenantAuthKey = []byte(k)
		}
	}
	if c.defaultBits > c.maxEntropyBits {
		c.defaultBits = c.maxEntropyBits
	}
	return c
}

// ---------------------------------------------------------------------------
// Entropy source.
// ---------------------------------------------------------------------------

type entropySource struct {
	extra *os.File // optional hardware RNG; nil = crypto/rand only
}

func newEntropySource(path string) (*entropySource, error) {
	es := &entropySource{}
	if path != "" {
		f, err := os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("open extra entropy %q: %w", path, err)
		}
		es.extra = f
	}
	// Fail-closed startup self-check: prove crypto/rand is usable before we
	// ever advertise ourselves as ready to mint secrets.
	probe := make([]byte, 32)
	if _, err := rand.Read(probe); err != nil {
		return nil, fmt.Errorf("crypto/rand unavailable: %w", err)
	}
	zero(probe)
	return es, nil
}

// bytes returns n cryptographically secure random bytes.
//
// When an extra hardware source is configured, its output is combined with the
// OS CSPRNG through HKDF (RFC 5869): PRK = Extract(salt=hardware, IKM=os) and
// out = Expand(PRK). Combining independent sources through an extractor cannot
// reduce the entropy of the stronger source, so mixing is always safe. If the
// configured hardware source errors, we fail closed rather than silently
// downgrading to OS-only (which would mask a misconfiguration).
func (es *entropySource) bytes(n int) ([]byte, error) {
	primary := make([]byte, n)
	if _, err := rand.Read(primary); err != nil {
		return nil, fmt.Errorf("read csprng: %w", err)
	}
	if es.extra == nil {
		return primary, nil
	}
	salt := make([]byte, n)
	if _, err := io.ReadFull(es.extra, salt); err != nil {
		zero(primary)
		return nil, fmt.Errorf("read extra entropy: %w", err)
	}
	defer zero(primary)
	defer zero(salt)
	return hkdfExpand(hkdfExtract(salt, primary), []byte("tenant-secret/v1"), n), nil
}

// hkdfExtract / hkdfExpand are the two halves of RFC 5869 HKDF over SHA-256,
// implemented with only the standard library to keep the enclave's attested
// measurement free of third-party dependencies.
func hkdfExtract(salt, ikm []byte) []byte {
	if len(salt) == 0 {
		salt = make([]byte, sha256.Size)
	}
	h := hmac.New(sha256.New, salt)
	h.Write(ikm)
	return h.Sum(nil)
}

func hkdfExpand(prk, info []byte, length int) []byte {
	var out, block []byte
	counter := byte(1)
	for len(out) < length {
		h := hmac.New(sha256.New, prk)
		h.Write(block)
		h.Write(info)
		h.Write([]byte{counter})
		block = h.Sum(nil)
		out = append(out, block...)
		counter++
	}
	return out[:length]
}

// ---------------------------------------------------------------------------
// Secret formats.
// ---------------------------------------------------------------------------

type format string

const (
	fmtHex       format = "hex"
	fmtBase64URL format = "base64url"
	fmtBase32    format = "base32"
	fmtAlnum     format = "alphanumeric"
	fmtPassword  format = "password"
)

const (
	alphabetAlnum    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	alphabetPassword = alphabetAlnum + "!#$%&*+-=?@^_~"
)

func parseFormat(s string) (format, bool) {
	switch format(s) {
	case fmtHex, fmtBase64URL, fmtBase32, fmtAlnum, fmtPassword:
		return format(s), true
	}
	return "", false
}

type secret struct {
	value       string
	entropyBits float64
}

// generate produces a secret carrying at least `bits` bits of entropy in the
// requested representation. For byte-oriented formats entropy is nbytes*8; for
// character formats each symbol contributes log2(len(alphabet)) bits and we
// round the character count up to meet the request.
func (es *entropySource) generate(f format, bits int) (secret, error) {
	switch f {
	case fmtHex, fmtBase64URL, fmtBase32:
		nbytes := (bits + 7) / 8
		b, err := es.bytes(nbytes)
		if err != nil {
			return secret{}, err
		}
		defer zero(b)
		var v string
		switch f {
		case fmtHex:
			v = hex.EncodeToString(b)
		case fmtBase64URL:
			v = base64.RawURLEncoding.EncodeToString(b)
		case fmtBase32:
			v = base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(b)
		}
		return secret{value: v, entropyBits: float64(nbytes * 8)}, nil

	case fmtAlnum, fmtPassword:
		alphabet := alphabetAlnum
		if f == fmtPassword {
			alphabet = alphabetPassword
		}
		perChar := math.Log2(float64(len(alphabet)))
		n := int(math.Ceil(float64(bits) / perChar))
		v, err := es.randomString(alphabet, n)
		if err != nil {
			return secret{}, err
		}
		return secret{value: v, entropyBits: float64(n) * perChar}, nil

	default:
		return secret{}, fmt.Errorf("unknown format %q", f)
	}
}

// randomString picks n symbols uniformly from alphabet using rejection
// sampling, which avoids the modulo bias a naive byte%len would introduce.
func (es *entropySource) randomString(alphabet string, n int) (string, error) {
	k := len(alphabet)
	limit := 256 - (256 % k) // reject bytes >= limit for a uniform distribution
	out := make([]byte, 0, n)
	for len(out) < n {
		batch, err := es.bytes((n-len(out))*2 + 8) // over-read to cut syscalls
		if err != nil {
			return "", err
		}
		for _, c := range batch {
			if int(c) < limit {
				out = append(out, alphabet[int(c)%k])
				if len(out) == n {
					break
				}
			}
		}
		zero(batch)
	}
	s := string(out)
	zero(out)
	return s, nil
}

// ---------------------------------------------------------------------------
// Per-tenant rate limiter (token bucket).
// ---------------------------------------------------------------------------
//
// In-memory and per-instance: fine for a single enclave. For horizontally
// scaled enclaves move this to shared state. The bucket map is unbounded in the
// number of distinct tenant IDs seen; add eviction if tenant IDs are attacker-
// controlled and high-cardinality.
type limiter struct {
	mu      sync.Mutex
	buckets map[string]*bucket
	rps     float64
	burst   float64
}

type bucket struct {
	tokens float64
	last   time.Time
}

func newLimiter(rps float64, burst int) *limiter {
	return &limiter{buckets: make(map[string]*bucket), rps: rps, burst: float64(burst)}
}

func (l *limiter) allow(key string) bool {
	if l.rps <= 0 {
		return true
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	b := l.buckets[key]
	if b == nil {
		b = &bucket{tokens: l.burst, last: now}
		l.buckets[key] = b
	}
	b.tokens += now.Sub(b.last).Seconds() * l.rps
	if b.tokens > l.burst {
		b.tokens = l.burst
	}
	b.last = now
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

// ---------------------------------------------------------------------------
// HTTP server.
// ---------------------------------------------------------------------------

type server struct {
	cfg     config
	entropy *entropySource
	rl      *limiter
	log     *log.Logger
}

type secretRequest struct {
	Bits   int    `json:"bits"`
	Format string `json:"format"`
}

type secretResponse struct {
	ID          string  `json:"id"`
	Tenant      string  `json:"tenant"`
	Secret      string  `json:"secret"`
	Format      string  `json:"format"`
	EntropyBits float64 `json:"entropy_bits"`
	CreatedAt   string  `json:"created_at"`
}

func (s *server) handleSecret(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	tenant, err := s.authenticateTenant(r)
	if err != nil {
		writeError(w, http.StatusUnauthorized, err.Error())
		return
	}
	if !s.rl.allow(tenant) {
		writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
		return
	}

	bits := s.cfg.defaultBits
	f := fmtBase64URL

	// Request parameters (no secret is ever carried in the request, so query
	// params are fine). Query first, JSON body overrides when present.
	if q := r.URL.Query().Get("bits"); q != "" {
		v, convErr := strconv.Atoi(q)
		if convErr != nil {
			writeError(w, http.StatusBadRequest, "bits must be an integer")
			return
		}
		bits = v
	}
	if q := r.URL.Query().Get("format"); q != "" {
		pf, ok := parseFormat(q)
		if !ok {
			writeError(w, http.StatusBadRequest, "unknown format")
			return
		}
		f = pf
	}
	if r.ContentLength != 0 {
		r.Body = http.MaxBytesReader(w, r.Body, 4<<10)
		dec := json.NewDecoder(r.Body)
		dec.DisallowUnknownFields()
		var req secretRequest
		if decErr := dec.Decode(&req); decErr != nil && !errors.Is(decErr, io.EOF) {
			writeError(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		if req.Bits != 0 {
			bits = req.Bits
		}
		if req.Format != "" {
			pf, ok := parseFormat(req.Format)
			if !ok {
				writeError(w, http.StatusBadRequest, "unknown format")
				return
			}
			f = pf
		}
	}

	if bits < 1 || bits > s.cfg.maxEntropyBits {
		writeError(w, http.StatusBadRequest,
			fmt.Sprintf("bits must be between 1 and %d", s.cfg.maxEntropyBits))
		return
	}

	sec, err := s.entropy.generate(f, bits)
	if err != nil {
		// Log the real cause internally; never leak internals to the caller.
		s.log.Printf("tenant=%s generate error: %v", tenant, err)
		writeError(w, http.StatusServiceUnavailable, "secret generation unavailable")
		return
	}

	id, err := newID()
	if err != nil {
		s.log.Printf("tenant=%s id error: %v", tenant, err)
		writeError(w, http.StatusServiceUnavailable, "secret generation unavailable")
		return
	}

	// Audit trail WITHOUT the secret.
	s.log.Printf("issued tenant=%s id=%s format=%s bits=%.0f", tenant, id, f, sec.entropyBits)

	writeJSON(w, http.StatusOK, secretResponse{
		ID:          id,
		Tenant:      tenant,
		Secret:      sec.value,
		Format:      string(f),
		EntropyBits: math.Round(sec.entropyBits*100) / 100,
		CreatedAt:   time.Now().UTC().Format(time.RFC3339),
	})
}

// authenticateTenant resolves and verifies the caller's tenant identity.
//
// PLACEHOLDER. It trusts X-Tenant-ID when accompanied by a valid HMAC in
// X-Tenant-Auth = hex(HMAC-SHA256(tenantAuthKey, tenantID)). That proves the
// caller knows a shared key but gives you neither per-tenant key separation nor
// rotation. In production replace with mTLS (derive tenant from the client cert
// SAN) or a JWT validated inside the enclave against a trusted issuer.
func (s *server) authenticateTenant(r *http.Request) (string, error) {
	tenant := strings.TrimSpace(r.Header.Get("X-Tenant-ID"))
	if tenant == "" || len(tenant) > 128 || !isSafeTenantID(tenant) {
		return "", errors.New("missing or invalid X-Tenant-ID")
	}
	if len(s.cfg.tenantAuthKey) == 0 {
		// Refuse rather than trust an unauthenticated header.
		return "", errors.New("tenant auth not configured")
	}
	provided, err := hex.DecodeString(strings.TrimSpace(r.Header.Get("X-Tenant-Auth")))
	if err != nil {
		return "", errors.New("invalid X-Tenant-Auth")
	}
	mac := hmac.New(sha256.New, s.cfg.tenantAuthKey)
	mac.Write([]byte(tenant))
	if !hmac.Equal(mac.Sum(nil), provided) { // constant-time comparison
		return "", errors.New("tenant authentication failed")
	}
	return tenant, nil
}

func (s *server) handleAttestation(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented,
		"attestation not wired: return your platform attestation document here "+
			"(Nitro NSM doc / SEV-SNP report / TDX quote) so clients can verify "+
			"the enclave before requesting secrets")
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	probe := make([]byte, 16)
	if _, err := rand.Read(probe); err != nil {
		writeError(w, http.StatusServiceUnavailable, "entropy unavailable")
		return
	}
	zero(probe)
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

func securityHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Content-Type", "application/json")
	h.Set("Cache-Control", "no-store") // never cache secret material
	h.Set("Pragma", "no-cache")
	h.Set("X-Content-Type-Options", "nosniff")
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	securityHeaders(w)
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false) // don't mangle & < > in symbol-heavy secrets
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// newID returns a random, NON-secret identifier for audit logs.
func newID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

// zero best-effort wipes a byte buffer. Note: once a secret is encoded to a Go
// string (unavoidable for a JSON response), that copy is immutable and cannot
// be reliably wiped and may be relocated by the GC. For stricter handling keep
// secrets as []byte end-to-end; an HTTP JSON API inherently serializes to text.
func zero(b []byte) {
	for i := range b {
		b[i] = 0
	}
}

func isSafeTenantID(s string) bool {
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '.', r == '-', r == '_':
		default:
			return false
		}
	}
	return true
}

func recoverMW(logger *log.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				logger.Printf("panic: %v", rec)
				writeError(w, http.StatusInternalServerError, "internal error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(k string, def float64) float64 {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil {
			return n
		}
	}
	return def
}

// ---------------------------------------------------------------------------
// main.
// ---------------------------------------------------------------------------

func main() {
	logger := log.New(os.Stderr, "", log.LstdFlags|log.LUTC)
	cfg := loadConfig()

	es, err := newEntropySource(cfg.extraEntropyPath)
	if err != nil {
		// Fail closed: do not start a secret generator that can't get entropy.
		logger.Fatalf("entropy init failed (refusing to start): %v", err)
	}

	srv := &server{
		cfg:     cfg,
		entropy: es,
		rl:      newLimiter(cfg.perTenantRPS, cfg.perTenantBurst),
		log:     logger,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/secret", srv.handleSecret)
	mux.HandleFunc("/v1/attestation", srv.handleAttestation)
	mux.HandleFunc("/healthz", srv.handleHealth)

	httpSrv := &http.Server{
		Addr:              cfg.addr,
		Handler:           recoverMW(logger, mux),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 16,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		logger.Printf("secret generator listening on %s (max_bits=%d extra_entropy=%t tenant_auth=%t)",
			cfg.addr, cfg.maxEntropyBits, cfg.extraEntropyPath != "", len(cfg.tenantAuthKey) > 0)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Fatalf("server error: %v", err)
		}
	}()

	<-ctx.Done()
	logger.Printf("shutdown signal received")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		logger.Printf("graceful shutdown failed: %v", err)
	}
	if es.extra != nil {
		_ = es.extra.Close()
	}
	logger.Printf("stopped")
}
