#ifndef SKYLARK_CODING_HPP
#define SKYLARK_CODING_HPP

namespace skylark { namespace ml {

template<typename T, typename LabelType>
void DummyCoding(El::Orientation orientation,
    El::Matrix<T> &Y, const El::Matrix<LabelType> &L,
    std::unordered_map<LabelType, El::Int> &coding,
    std::vector<LabelType> &rcoding,
    T pval = T(1.0), T nval = T(-1.0)) {

    // TODO verify that height or width of L is 1.

    // Figure out the coding
    coding.clear();
    rcoding.clear();
    El::Int num = 0;
    const LabelType *l = L.LockedBuffer();
    for(El::Int i = 0; i < L.Height() * L.Width(); i++) {
        const LabelType &label = l[i];
        if (coding.count(label) == 0) {
            coding[label] = num;
            rcoding.push_back(label);
            num++;
        }
    }

    // Apply mapping
    if (orientation == El::NORMAL) {
        Y.Resize(L.Height() * L.Width(), num);
        El::EntrywiseMap(Y,
            std::function<T(const T &)>([nval] (const T &x) { return nval; }));
        for(El::Int i = 0; i < L.Height() * L.Width(); i++)
            Y.Set(i, coding[l[i]], pval);

    } else {
        Y.Resize(num, L.Height() * L.Width());
        El::EntrywiseMap(Y,
            std::function<T(const T &)>([nval] (const T &x) { return nval; }));
        for(El::Int i = 0; i < L.Height() * L.Width(); i++)
            Y.Set(coding[l[i]], i, pval);

    }
}

template<typename T, typename LabelType>
void DummyCoding(El::Orientation orientation,
    El::AbstractDistMatrix<T> &Y, const El::AbstractDistMatrix<LabelType> &L,
    std::unordered_map<LabelType, El::Int> &coding,
    std::vector<LabelType> &rcoding,
    T pval = T(1.0), T nval = T(-1.0)) {

    // TODO verify that height or width of L is 1.

    // Figure out the coding
    El::DistMatrix<LabelType, El::STAR, El::STAR> Lc = L;
    coding.clear();
    rcoding.clear();
    El::Int num = 0;
    const LabelType *l = Lc.LockedBuffer();
    for(El::Int i = 0; i < Lc.Height() * Lc.Width(); i++) {
        const LabelType &label = l[i];
        if (coding.count(label) == 0) {
            coding[label] = num;
            rcoding.push_back(label);
            num++;
        }
    }

    // Apply mapping
    if (orientation == El::NORMAL) {
        El::DistMatrix<T, El::VC, El::STAR> Yc(L.Height() * L.Width(), num);
        El::EntrywiseMap(Yc,
            std::function<T(const T &)>([nval] (const T &x) { return nval; }));
        El::Matrix<T> &Yl = Yc.Matrix();
        for(El::Int i = 0; i < Yl.Height(); i++)
            Yl.Set(i, coding[l[Yc.GlobalRow(i)]], pval);

        El::Copy(Yc, Y);

    } else {
        El::DistMatrix<T, El::STAR, El::VC> Yc(num, L.Height() * L.Width());
        El::EntrywiseMap(Yc,
            std::function<T(const T &)>([nval] (const T &x) { return nval; }));
        El::Matrix<T> &Yl = Yc.Matrix();
        for(El::Int i = 0; i < Yl.Width(); i++)
            Yl.Set(coding[l[Yc.GlobalCol(i)]], i, pval);

        El::Copy(Yc, Y);
    }
}

template<typename T, typename LabelType>
void DummyDecode(El::Orientation orientation,
    const El::Matrix<T> &Y, El::Matrix<LabelType> &L,
    const std::vector<LabelType> &rcoding,
    bool maxidx = true) {

    if (orientation == El::ADJOINT) {
        L.Resize(1, Y.Width());
        LabelType *l = L.Buffer();
        const T *y = Y.LockedBuffer();
        El::Int ld = Y.LDim();
        for(El::Int j = 0; j < Y.Width(); j++) {
            El::Int idx = 0;
            for(El::Int i = 1; i < Y.Height(); i++)
                if ((maxidx && y[j * ld + i] > y[j * ld + idx]) ||
                    (!maxidx && y[j * ld + i] < y[j * ld + idx]))
                    idx = i;
            l[j] = rcoding[idx];
        }

    }

    // TODO: normal
}

template<typename T, typename LabelType>
void DummyDecode(El::Orientation orientation,
    const El::AbstractDistMatrix<T> &Y, El::AbstractDistMatrix<LabelType> &L,
    const std::vector<LabelType> &rcoding,
    bool maxidx = true) {

    if (orientation == El::ADJOINT) {
        El::DistMatrix<T, El::STAR, El::VR> Yc = Y;
        El::DistMatrix<LabelType, El::STAR, El::VR> Lc(1, Y.Width());

        DummyDecode(orientation, Yc.Matrix(), Lc.Matrix(), rcoding, maxidx);

        El::Copy(Lc, L);
    }

    if (orientation == El::NORMAL) {
        El::DistMatrix<T, El::VC, El::STAR> Yc = Y;
        El::DistMatrix<LabelType, El::VC, El::STAR> Lc(Y.Height(), 1);

        DummyDecode(orientation, Yc.Matrix(), Lc.Matrix(), rcoding, maxidx);

        El::Copy(Lc, L);
    }
}

} } // namespace skylark::ml

#endif // SKYLARK_CODING_HPP
