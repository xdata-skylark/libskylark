
import randgen

base = 123
num_samples = 5
seed = 456


print
print "normal_distribution"
rng = randgen.normal_distribution_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample
print

print
print "uniform_int_distribution"
rng = randgen.uniform_int_distribution_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample
print

print
print "uniform_real_distribution"
rng = randgen.uniform_real_distribution_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample
print

print
print "cauchy_distribution"
rng = randgen.cauchy_distribution_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample
print

print
print "exponential_distribution"
rng = randgen.exponential_distribution_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample
print

print
print "standard_levy_distribution_t"
rng = randgen.standard_levy_distribution_t_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample
print

print
print "rademacher_distribution_t"
rng = randgen.rademacher_distribution_t_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample
print

