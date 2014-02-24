code_filename = 'test_randgen.py'

header_code = '''
import randgen

base = 123
num_samples = 5
seed = 456

'''


test_template = '''
print
print "%(distribution_type)s"
rng = randgen.%(distribution_type)s_array(base, seed)
samples = [rng[i] for i in range(num_samples)]
for sample in samples:
    print sample
print
'''

distribution_types = [{'distribution_type' : 'normal_distribution'},
                      {'distribution_type' : 'uniform_int_distribution'},
                      {'distribution_type' : 'uniform_real_distribution'},
                      {'distribution_type' : 'cauchy_distribution'},
                      {'distribution_type' : 'exponential_distribution'},
                      {'distribution_type' : 'standard_levy_distribution_t'},
                      {'distribution_type' : 'rademacher_distribution_t'}]

tests_code = ''
for distribution_type in distribution_types:
    test_code = test_template % distribution_type
    tests_code += test_code

code = header_code + tests_code

f = open(code_filename, 'w')
print >> f, code
f.close()

