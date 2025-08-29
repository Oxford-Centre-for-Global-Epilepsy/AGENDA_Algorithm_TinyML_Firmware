#ifndef AVG_POOL_H
#define AVG_POOL_H

class AvgPool {
public:
    AvgPool(int dim);

    void reset();
    void add(const float* vec);
    void finalize(float* out);

private:
    int dim_;
    float* sum_;
    int count_;
};

#endif