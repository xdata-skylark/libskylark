#ifndef LAPACK_PROTOTYPES_HPP
#define LAPACK_PROTOTYPES_HPP

extern "C" {
    void daxpy_ (int*,    /* N */
                 double*, /* alpha */
                 double*, /* X */
                 int*,    /* INCX */
                 double*, /* Y */
                 int*);   /* INCY */

    void dgemv_ (char*,   /* Transpose A or not */
                 int*,    /* M */
                 int*,    /* N */
                 double*, /* alpha */
                 double*, /* A */
                 int*,    /* LDA */
                 double*, /* X */
                 int*,    /* Increment for elements of X */
                 double*, /* beta */
                 double*, /* Y */
                 int*);   /* Increment for elements of Y */

    void dgelss_ (int*,   /* M */
                  int*,   /* N */
                  int*,   /* Number of right-hand sides */
                  double*,/* A */
                  int*,   /* LDA */
                  double*,/* B */
                  int*,   /* LDB */
                  double*,/* S -- array of singular values */
                  double*,/* RCOND --- used to determine rank of A */
                  int*,   /* Rank of A */
                  double*,/* Work */
                  int*,   /* LDWORK */
                  int*);  /* Result */

    void dgemm_ (char*,   /* transpose A */
                 char*,   /* transpose B */
                 int*,    /* number of rows in C */
                 int*,    /* number of columns in C */
                 int*,    /* number of columns in A if 'n' is used */
                 double*, /* alpha */
                 double*, /* A */
                 int*,    /* LDA */
                 double*, /* B */
                 int*,    /* LDB */
                 double*, /* beta */
                 double*, /* C */
                 int*);   /* LDC */

    void dsyrk_ (char*,   /* UPLO -- upper or lower triangular */
                 char*,   /* 'T'=> A'A, 'N'==>AA' */
                 int*,    /* number of rows/columns in C */
                 int*,    /* 'N' ==> number of columns in A */
                 /* 'T' ==> number of rows of A */
                 double*, /* alpha */
                 double*, /* A */
                 int*,    /* LDA */
                 double*, /* beta */
                 double*, /* C */
                 int*);   /* LDC */

    void dpotrf_(char*,   /* uplo */
                 int*,    /* N */
                 double*, /* A */
                 int*,    /* lda */
                 int*);   /* info */

    void dtrsm_(char*,    /* side */
                char*,    /* uplo */
                char*,    /* transA */
                char*,    /* diag */
                int*,     /* M */
                int*,     /* N */
                double*,  /* alpha */
                double*,  /* A */
                int*,     /* lda */
                double*,  /* B */
                int*);    /* ldb */

    double dnrm2_(int*,    /* N */
                  double*, /* x */
                  int*);   /* incX */

    double ddot_(int*,    /* N */
                 double*, /* x */
                 int*,    /* incX */
                 double*, /* y */
                 int*);   /* incY */

    void dcopy_ (int*,     /* M */
                 double*,  /* src vector */
                 int*,     /* increments to source */
                 double*,  /* dst vector */
                 int*);    /* increments to dst */

    void dgeqrf_ (int*,     /* M */
                  int*,     /* N */
                  double*,  /* A */
                  int*,     /* LDA */
                  double*,  /* tau */
                  double*,  /* work */
                  int*,     /* len of work */
                  int*);    /* info */

    void dgeqr2_ (int*,     /* M */
                  int*,     /* N */
                  double*,  /* A */
                  int*,     /* LDA */
                  double*,  /* tau */
                  double*,  /* work */
                  int*);    /* info */

    void dtorg_ (double*,
                 double*,
                 double*,
                 double*);

    void drot_ (int*,
                double*,
                int*,
                double*,
                int*,
                double*,
                double*);

    void dlasr_ (char*,
                 char*,
                 char*,
                 int*,
                 int*,
                 double*,
                 double*,
                 double*,
                 int*);

    void dormqr_ (char*,        /* side */
                  char*,        /* trans */
                  int*,         /* M */
                  int*,         /* N */
                  int*,         /* K, the number of elementary reflectors */
                  double*,      /* A */
                  int*,         /* LDA */
                  double*,      /* TAU */
                  double*,      /* C */
                  int*,         /* LDC */
                  double*,      /* work */
                  int*,         /* lwork */
                  int*);        /* info */

    void dlarft_ (char*,   /* direction */
                  char*,   /* store format */
                  int*,    /* order of the block reflector */
                  int*,    /* number of block reflectors */
                  double*, /* store for the block reflectors */
                  int*,    /* leading dimension of the store */
                  double*, /* tau --- scalar factors */
                  double*, /* t --- to store the triangular block reflector */
                  int*);   /* leading dimension of t */

    void dlarfb_ (char*,   /* side  */
                  char*,   /* trans */
                  char*,   /* direction */
                  char*,   /* column or row major storage */
                  int*,    /* M */
                  int*,    /* N */
                  int*,    /* K */
                  double*, /* V */
                  int*,    /* LDV */
                  double*, /* t */
                  int*,    /* LDT */
                  double*, /* C */
                  int*,    /* LDC */
                  double*, /* work */
                  int*);   /* LDWORK */

    void dorgqr_ (int*,    /* M */
                  int*,    /* N */
                  int*,    /* K */
                  double*,  /* A */
                  int*,     /* LDA */
                  double*,  /* tau */
                  double*,  /* work */
                  int*,     /* len of work */
                  int*);    /* info */

    void dlaswp_ (int*,    /* Number of columns in A */
                  double*, /* Matrix A */
                  int*,    /* LDA */
                  int*,    /* First pivot element to be accessed */
                  int*,    /* Last pivot element to be accessed */
                  int*,    /* Integer array of pivots */
                  int*);   /* increments to pivot array */
}

#endif // LAPACK_PROTOTYPES_HPP
