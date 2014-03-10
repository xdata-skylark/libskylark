code_filename = 'randgen.pyx'

imports_code = '''
from cython.operator cimport dereference as deref
from utility cimport *

'''

array_template = '''
cdef class %(distribution_type)s_array:
   cdef random_samples_array_t[%(distribution_type)s[%(value_type)s] ] *thisptr
   cdef %(distribution_type)s[%(value_type)s] *distribution
   
   def __cinit__(self, size_t base, int seed, %%(parameter_signature_string)s):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new %(distribution_type)s[%(value_type)s](%%(parameter_value_string)s)
       self.thisptr  = new random_samples_array_t[%(distribution_type)s[%(value_type)s] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return <%(value_type)s>self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution

''' 


distributions = [
    {'distribution_type' : 'normal_distribution',
     'value_type' : 'double', 
     'parameters' : [{'name': 'mean_arg',    'default_value': '0.0'}, 
                     {'name': 'sigma_arg',   'default_value': '1.0'}]},
    
    {'distribution_type' : 'uniform_int_distribution',
     'value_type' : 'int', 
     'parameters' : [{'name': 'min_arg',    'default_value': '0'}, 
                     {'name': 'max_arg',    'default_value': '10'}]},

    {'distribution_type' : 'uniform_real_distribution',
     'value_type' : 'double', 
     'parameters' : [{'name': 'min_arg',    'default_value': '0.0'}, 
                     {'name': 'max_arg',    'default_value': '1.0'}]},

    {'distribution_type' : 'cauchy_distribution',
     'value_type' : 'double', 
     'parameters' : [{'name': 'median_arg',    'default_value': '0.0'}, 
                     {'name': 'sigma_arg',     'default_value': '1.0'}]},

    {'distribution_type' : 'exponential_distribution',
     'value_type' : 'double', 
     'parameters' : [{'name': 'lambda_arg',    'default_value': '1.0'}]},

    {'distribution_type' : 'standard_levy_distribution_t',
     'value_type' : 'double',
     'parameters' : []},
    
    {'distribution_type' : 'rademacher_distribution_t',
     'value_type' : 'int', 
     'parameters' : []}       
]


definitions_code = ''    
for distribution in distributions:
    gen_template = array_template % distribution
    parameters = distribution['parameters']
    parameter_signature_string = ''
    parameter_value_string = ''
    for parameter in parameters:
        parameter_signature_string += '%(name)s = %(default_value)s, ' % parameter
        parameter_value_string     += '%(name)s, ' % parameter
    definition_code = gen_template % {'parameter_signature_string' : parameter_signature_string[:-2],
                           'parameter_value_string'     : parameter_value_string[:-2]}
    definitions_code += definition_code

code = imports_code + definitions_code

f = open(code_filename, 'w')
print >> f, code
f.close()
