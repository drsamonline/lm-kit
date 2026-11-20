/*
 * lm.c — a small inference kernel for the models lm-train.py produces.
 *
 *   make            builds ./lm
 *   ./lm model.lmk "Job postings from 185 sources"
 *
 * What it is
 * ----------
 * One architecture, one file, no dependencies beyond libc and libm. It loads the .lmk
 * that lm-export-c.py writes, dequantises the int8 weights once, and runs the same
 * forward pass the trainer ran: RMSNorm, RoPE, causal attention, SwiGLU, tied output.
 *
 * Correct before fast
 * -------------------
 * The weights are stored int8 to keep the file small, and expanded to float on load. The
 * inner loop is plain float arithmetic, not an integer dot product, because the first job
 * is to produce the same tokens as PyTorch. Making the matmul integer is a separate change
 * with its own benchmark — and a benchmark that does not compare against a correctly
 * configured baseline is how 1.17x becomes "7.5x".
 *
 * Licence: PolyForm Noncommercial 1.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXD 1024
#define MAXFF 4096
#define MAXLAYERS 32
#define MAXTOK 4096

/* ------------------------------------------------------------------ model */

typedef struct {
    char name[64];
    int ndim;
    uint32_t dims[4];
    int dtype;          /* 0 = f32, 1 = int8 */
    uint32_t nbytes;
    unsigned char *raw;
} Tensor;

typedef struct {
    uint32_t vocab, dim, layers, heads, ff, seq;
    float *emb, *head;
    float *qkv[MAXLAYERS], *proj[MAXLAYERS];
    float *w1[MAXLAYERS], *w3[MAXLAYERS], *w2[MAXLAYERS];
    float *n1[MAXLAYERS], *n2[MAXLAYERS], *nfinal;
    int ok;
} Model;

static uint32_t rd_u32(FILE *f) {
    uint32_t v = 0;
    if (fread(&v, 4, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
    return v; /* little-endian host assumed; every target we ship is */
}

static Tensor *load_tensors(const char *path, uint32_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "LMK1", 4) != 0) {
        fprintf(stderr, "not an .lmk file\n"); exit(1);
    }
    uint32_t version = rd_u32(f), n = rd_u32(f);
    if (version != 1) { fprintf(stderr, "version %u unsupported\n", version); exit(1); }
    Tensor *t = calloc(n, sizeof(Tensor));
    for (uint32_t i = 0; i < n; i++) {
        uint8_t nl = 0;
        if (fread(&nl, 1, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
        if (fread(t[i].name, 1, nl, f) != nl) { fprintf(stderr, "short read\n"); exit(1); }
        t[i].name[nl] = 0;
        uint8_t nd = 0;
        if (fread(&nd, 1, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
        t[i].ndim = nd;
        for (int d = 0; d < nd; d++) t[i].dims[d] = rd_u32(f);
        uint8_t dt = 0;
        if (fread(&dt, 1, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
        t[i].dtype = dt;
        t[i].nbytes = rd_u32(f);
        t[i].raw = malloc(t[i].nbytes);
        if (fread(t[i].raw, 1, t[i].nbytes, f) != t[i].nbytes) {
            fprintf(stderr, "short read on %s\n", t[i].name); exit(1);
        }
    }
    fclose(f);
    *count = n;
    return t;
}

static Tensor *find(Tensor *t, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (strcmp(t[i].name, name) == 0) return &t[i];
    return NULL;
}

/* int8 with a per-row scale -> float, one row at a time */
static float *dequant_rows(Tensor *w, Tensor *scale, int rows, int cols) {
    float *out = malloc(sizeof(float) * rows * cols);
    const int8_t *q = (const int8_t *)w->raw;
    const float *s = (const float *)scale->raw;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            out[r * cols + c] = (float)q[r * cols + c] * s[r];
    return out;
}

static float *dequant_plain(Tensor *w) {
    float *out = malloc(w->nbytes);
    memcpy(out, w->raw, w->nbytes);
    return out;
}

static float *get_mat(Tensor *t, uint32_t n, const char *base, int rows, int cols) {
    char sname[96];
    Tensor *w = find(t, n, base);
    if (!w) { fprintf(stderr, "missing tensor %s\n", base); exit(1); }
    if (w->dtype == 0) return dequant_plain(w);
    snprintf(sname, sizeof sname, "%s.scale", base);
    Tensor *s = find(t, n, sname);
    if (!s) { fprintf(stderr, "missing scale for %s\n", base); exit(1); }
    return dequant_rows(w, s, rows, cols);
}

static float *get_vec(Tensor *t, uint32_t n, const char *base) {
    Tensor *w = find(t, n, base);
    if (!w) { fprintf(stderr, "missing tensor %s\n", base); exit(1); }
    return dequant_plain(w);
}

static void load_model(Model *m, const char *path) {
    uint32_t n = 0;
    Tensor *t = load_tensors(path, &n);
    Tensor *emb = find(t, n, "emb/weight");
    if (!emb) { fprintf(stderr, "no emb/weight\n"); exit(1); }
    m->vocab = emb->dims[0];
    m->dim   = emb->dims[1];

    /* infer layers by counting qkv tensors */
    uint32_t layers = 0;
    while (1) {
        char name[64];
        snprintf(name, sizeof name, "blocks/%u/qkv/weight", layers);
        if (!find(t, n, name)) break;
        layers++;
    }
    m->layers = layers;
    if (layers == 0 || layers > MAXLAYERS) { fprintf(stderr, "bad layer count\n"); exit(1); }

    /* heads: from the exported config, which is not in the file — derive from dim/32 */
    /* the shape travels in the file; guessing it silently miscomputes */
    Tensor *shape = find(t, n, "config/shape");
    if (!shape || shape->nbytes < 24) {
        fprintf(stderr, "no config/shape in %s — re-export with a current lm-export-c.py\n", path);
        exit(1);
    }
    const int32_t *sh = (const int32_t *)shape->raw;
    m->vocab  = sh[0];
    m->dim    = sh[1];
    m->layers = sh[2];
    m->heads  = sh[3];
    m->ff     = sh[4];
    m->seq    = sh[5];

    char name[64];
    snprintf(name, sizeof name, "blocks/0/w1/weight");
    Tensor *w1 = find(t, n, name);
    if (w1) m->ff = w1->dims[0];

    m->emb = get_mat(t, n, "emb/weight", m->vocab, m->dim);
    m->head = m->emb; /* tied */

    for (uint32_t l = 0; l < layers; l++) {
        snprintf(name, sizeof name, "blocks/%u/qkv/weight", l);
        m->qkv[l]  = get_mat(t, n, name, 3 * m->dim, m->dim);
        snprintf(name, sizeof name, "blocks/%u/proj/weight", l);
        m->proj[l] = get_mat(t, n, name, m->dim, m->dim);
        snprintf(name, sizeof name, "blocks/%u/w1/weight", l);
        m->w1[l]   = get_mat(t, n, name, m->ff, m->dim);
        snprintf(name, sizeof name, "blocks/%u/w3/weight", l);
        m->w3[l]   = get_mat(t, n, name, m->ff, m->dim);
        snprintf(name, sizeof name, "blocks/%u/w2/weight", l);
        m->w2[l]   = get_mat(t, n, name, m->dim, m->ff);
        snprintf(name, sizeof name, "blocks/%u/n1/w", l);
        m->n1[l]   = get_vec(t, n, name);
        snprintf(name, sizeof name, "blocks/%u/n2/w", l);
        m->n2[l]   = get_vec(t, n, name);
    }
    m->nfinal = get_vec(t, n, "norm/w");
    m->ok = 1;
    for (uint32_t i = 0; i < n; i++) free(t[i].raw);
    free(t);
}


/* ------------------------------------------------------------- tokenizer */

/*
 * Byte-level BPE. The merges travel inside the .lmk as an int32 [n][2] tensor, so the
 * kernel tokenises the same way the trainer did without a second file or a runtime.
 */

#define HBITS 17
#define HSIZE (1 << HBITS)

typedef struct {
    int n;              /* number of merges */
    int32_t *pairs;     /* n * 2 */
    int32_t *rank_key;  /* open-addressed: a * 8192 + b, -1 when empty */
    int32_t *rank_val;
    unsigned char **expand;  /* id -> bytes */
    int *expand_len;
    int vocab;
} Tok;

static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

static void tok_build(Tok *t, Tensor *m, int vocab) {
    t->n = m ? (int)m->dims[0] : 0;
    t->pairs = m ? (int32_t *)m->raw : NULL;
    t->vocab = vocab;
    t->rank_key = malloc(sizeof(int32_t) * HSIZE);
    t->rank_val = malloc(sizeof(int32_t) * HSIZE);
    for (int i = 0; i < HSIZE; i++) t->rank_key[i] = -1;
    for (int i = 0; i < t->n; i++) {
        int32_t a = t->pairs[2 * i], b = t->pairs[2 * i + 1];
        uint32_t h = hash32((uint32_t)(a * 8192 + b)) & (HSIZE - 1);
        while (t->rank_key[h] != -1) h = (h + 1) & (HSIZE - 1);
        t->rank_key[h] = a * 8192 + b;
        t->rank_val[h] = i;
    }
    /* resolve every merge to bytes once, in order, so decode is a copy */
    t->expand = calloc(vocab, sizeof(unsigned char *));
    t->expand_len = calloc(vocab, sizeof(int));
    for (int i = 0; i < t->n; i++) {
        int32_t a = t->pairs[2 * i], b = t->pairs[2 * i + 1];
        int la = a < 256 ? 1 : t->expand_len[a];
        int lb = b < 256 ? 1 : t->expand_len[b];
        unsigned char *e = malloc(la + lb);
        if (a < 256) e[0] = (unsigned char)a; else memcpy(e, t->expand[a], la);
        if (b < 256) e[la] = (unsigned char)b; else memcpy(e + la, t->expand[b], lb);
        t->expand[256 + i] = e;
        t->expand_len[256 + i] = la + lb;
    }
}

static int tok_rank(Tok *t, int32_t a, int32_t b) {
    if (!t->n) return -1;
    uint32_t h = hash32((uint32_t)(a * 8192 + b)) & (HSIZE - 1);
    while (t->rank_key[h] != -1) {
        if (t->rank_key[h] == a * 8192 + b) return t->rank_val[h];
        h = (h + 1) & (HSIZE - 1);
    }
    return -1;
}

/* encode one word (no spaces) into out; returns the count */
static int tok_encode_word(Tok *t, const unsigned char *w, int len, int *out) {
    int syms[256], n = len;
    if (len > 256) return 0;
    for (int i = 0; i < len; i++) syms[i] = w[i];
    while (n > 1) {
        int best = -1, bi = -1;
        for (int i = 0; i < n - 1; i++) {
            int r = tok_rank(t, syms[i], syms[i + 1]);
            if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (best < 0) break;
        syms[bi] = 256 + best;
        for (int i = bi + 1; i < n - 1; i++) syms[i] = syms[i + 1];
        n--;
    }
    for (int i = 0; i < n; i++) out[i] = syms[i];
    return n;
}

/* encode a whole string: whitespace-split words, a space between them, like the trainer */
static int tok_encode(Tok *t, const char *text, int *out) {
    int n = 0;
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (!*p) break;
        const unsigned char *start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
        n += tok_encode_word(t, start, (int)(p - start), out + n);
        out[n++] = 32;
    }
    if (n > 0) n--;
    return n;
}

/* ------------------------------------------------------------------ maths */

static void rmsnorm(const float *x, const float *w, float *out, int d) {
    float ms = 0.0f;
    for (int i = 0; i < d; i++) ms += x[i] * x[i];
    ms = ms / d + 1e-5f;
    float inv = 1.0f / sqrtf(ms);
    for (int i = 0; i < d; i++) out[i] = w[i] * x[i] * inv;
}

static void matvec(const float *W, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row = W + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += row[c] * x[c];
        y[r] = acc;
    }
}

static void rope(float *v, int nheads, int dh, int pos) {
    for (int h = 0; h < nheads; h++) {
        float *x = v + (size_t)h * dh;
        for (int j = 0; j < dh / 2; j++) {
            float freq = powf(10000.0f, -2.0f * j / (float)dh);
            float a = pos * freq;
            float c = cosf(a), s = sinf(a);
            float x0 = x[2 * j], x1 = x[2 * j + 1];
            x[2 * j]     = x0 * c - x1 * s;
            x[2 * j + 1] = x0 * s + x1 * c;
        }
    }
}

static void silu_mul(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float x = a[i];
        out[i] = (x / (1.0f + expf(-x))) * b[i];
    }
}

/* forward one token at position pos, with the KV cache kept by the caller */
typedef struct {
    float *k[MAXLAYERS], *v[MAXLAYERS]; /* seq x dim each */
    /* scratch, owned by the caller: forward() keeps no state of its own, so two
       models can run in one process and the function is safe to call from a thread
       that owns its cache */
    float x[MAXD], n1[MAXD], n2[MAXD], qkv[3 * MAXD], att[MAXD];
    float h1[MAXFF], h3[MAXFF], hm[MAXFF];
    float scores[MAXD];
} Cache;

static void forward(Model *m, int token, int pos, Cache *c, float *logits) {
    int d = m->dim, nh = m->heads, dh = d / nh;
    if (pos < 0 || (uint32_t)pos >= m->seq) {
        fprintf(stderr, "context full at %d (seq %u)\n", pos, m->seq);
        exit(3);
    }
    float *x = c->x, *n1 = c->n1, *n2 = c->n2, *qkv = c->qkv, *att = c->att;
    float *h1 = c->h1, *h3 = c->h3, *hm = c->hm, *scores = c->scores;

    for (int i = 0; i < d; i++) x[i] = m->emb[(size_t)token * d + i];

    for (uint32_t l = 0; l < m->layers; l++) {
        rmsnorm(x, m->n1[l], n1, d);
        matvec(m->qkv[l], n1, qkv, 3 * d, d);
        float *q = qkv, *k = qkv + d, *v = qkv + 2 * d;
        rope(q, nh, dh, pos);
        rope(k, nh, dh, pos);
        memcpy(c->k[l] + (size_t)pos * d, k, sizeof(float) * d);
        memcpy(c->v[l] + (size_t)pos * d, v, sizeof(float) * d);

        for (int h = 0; h < nh; h++) {
            float *qh = q + h * dh;
            float best = -1e30f;
            for (int p = 0; p <= pos; p++) {
                const float *kh = c->k[l] + (size_t)p * d + h * dh;
                float s = 0.0f;
                for (int j = 0; j < dh; j++) s += qh[j] * kh[j];
                scores[p] = s / sqrtf((float)dh);
                if (scores[p] > best) best = scores[p];
            }
            float sum = 0.0f;
            for (int p = 0; p <= pos; p++) { scores[p] = expf(scores[p] - best); sum += scores[p]; }
            for (int p = 0; p <= pos; p++) scores[p] /= sum;
            for (int j = 0; j < dh; j++) {
                float acc = 0.0f;
                for (int p = 0; p <= pos; p++)
                    acc += scores[p] * c->v[l][(size_t)p * d + h * dh + j];
                att[h * dh + j] = acc;
            }
        }
        matvec(m->proj[l], att, n2, d, d);
        for (int i = 0; i < d; i++) x[i] += n2[i];

        rmsnorm(x, m->n2[l], n2, d);
        matvec(m->w1[l], n2, h1, m->ff, d);
        matvec(m->w3[l], n2, h3, m->ff, d);
        silu_mul(h1, h3, hm, m->ff);
        matvec(m->w2[l], hm, n2, d, m->ff);
        for (int i = 0; i < d; i++) x[i] += n2[i];
    }

    rmsnorm(x, m->nfinal, n1, d);
    matvec(m->head, n1, logits, m->vocab, d);
}

/* ------------------------------------------------------------------ main */

#define BOS 257
#define EOS 258

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.lmk [prompt] [max_tokens]\n", argv[0]);
        return 2;
    }
    Model m;
    memset(&m, 0, sizeof m);
    load_model(&m, argv[1]);

    const char *prompt = argc > 2 ? argv[2] : "Job postings from 185 sources";
    int max_new = argc > 3 ? atoi(argv[3]) : 60;

    Tok tok;
    memset(&tok, 0, sizeof tok);
    {
        uint32_t nn = 0;
        Tensor *tt = load_tensors(argv[1], &nn);
        tok_build(&tok, find(tt, nn, "tokenizer/merges"), (int)m.vocab);
        free(tt);
    }

    char buf[2048];
    snprintf(buf, sizeof buf, "IN: %s\nOUT: ", prompt);
    int ids[MAXTOK], n = 0;
    ids[n++] = BOS;
    if (tok.n) {
        int room = MAXTOK - 1 - n;
        int got = tok_encode(&tok, buf, ids + n);
        if (got > room) {
            fprintf(stderr, "prompt too long: %d tokens, %d room\n", got, room);
            return 3;
        }
        n += got;
    } else {
        for (const char *p = buf; *p && n < MAXTOK - 1; p++) ids[n++] = (unsigned char)*p;
    }

    Cache c;
    for (uint32_t l = 0; l < m.layers; l++) {
        c.k[l] = calloc((size_t)m.seq * m.dim, sizeof(float));
        c.v[l] = calloc((size_t)m.seq * m.dim, sizeof(float));
    }

    static float logits[8192];
    int pos = 0;
    for (int i = 0; i < n; i++, pos++) forward(&m, ids[i], pos, &c, logits);

    fputc('\n', stdout);
    for (int g = 0; g < max_new; g++, pos++) {
        if ((uint32_t)pos >= m.seq) {
            fprintf(stderr, "\n[stopped at the %u-token context]\n", m.seq);
            break;
        }
        int best = 0;
        for (int i = 1; i < (int)m.vocab; i++) if (logits[i] > logits[best]) best = i;
        if (best == EOS) break;
        if (tok.n && best >= 256) {
            fwrite(tok.expand[best], 1, tok.expand_len[best], stdout);
        } else if (best < 256) {
            fputc(best, stdout);
        } else {
            printf("<%d>", best);
        }
        fflush(stdout);
        forward(&m, best, pos, &c, logits);
    }
    fputc('\n', stdout);

    for (uint32_t l = 0; l < m.layers; l++) { free(c.k[l]); free(c.v[l]); }
    return 0;
}
