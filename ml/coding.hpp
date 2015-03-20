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
    for(int i = 0; i < L.Height() * L.Width(); i++) {
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
            std::function<T(T)>([nval] (T x) { return nval; }));
        T *y = Y.Buffer();
        for(int i = 0; i < L.Height() * L.Width(); i++)
            Y.Set(i, coding[l[i]], pval);

    } else {
        Y.Resize(num, L.Height() * L.Width());
        El::EntrywiseMap(Y,
            std::function<T(T)>([nval] (T x) { return nval; }));
        T *y = Y.Buffer();
        for(int i = 0; i < L.Height() * L.Width(); i++)
            Y.Set(coding[l[i]], i, pval);

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
        const double *y = Y.LockedBuffer();
        El::Int ld = Y.LDim();
        for(int j = 0; j < Y.Width(); j++) {
            int idx = 0;
            for(int i = 1; i < Y.Height(); i++)
                if ((maxidx && y[j * ld + i] > y[j * ld + idx]) ||
                    (!maxidx && y[j * ld + i] < y[j * ld + idx]))
                    idx = i;
            l[j] = rcoding[idx];
        }

    }

}

} } // namespace skylark::ml

#endif // SKYLARK_CODING_HPP
