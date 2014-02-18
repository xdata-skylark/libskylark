import randgen

base = 123
num_samples = 5
seed = 456

rng = randgen.normal_distribution_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample


