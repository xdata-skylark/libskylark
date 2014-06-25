.. highlight:: rst

Base Layer
**********

This layer provides the foundation of the entire libSkylark library.

Random Values
-------------

Random values are generated in a lazy fashion using
`Random123 library <http://www.deshawresearch.com/resources_random123.html>`_.
The access to the random streams is wrapped in two classes, which abstracts random
number arrays.

.. cpp:type:: struct skylark::utility::random_samples_array_t<Distribution>

    .. cpp:function:: random_samples_array_t(size_t base, size_t size, int seed, Distribution& distribution)

        Random-access array of samples drawn from a distribution starting from
        `base` up to a total of `size` samples. The `distribution` determines
        how samples are drawn.

    .. cpp:function:: random_samples_array_t(const random_samples_array_t& other)

    .. cpp:function:: random_samples_array_t& operator=(const random_samples_array_t& other)

    **Accessors**

    .. cpp:function:: value_type operator[](size_t index) const

        Returns the random value (following the specified distribution) at
        position `index` in the random stream.

libSkylark Context
------------------

The library provides and tracks the state of the random number streams in the `context`
class. Any use of functionality that generates random numbers will need to provide a context
object.

.. cpp:type:: struct skylark::base::context_t

    **Constructor**

    .. cpp:function:: context_t (int seed, int counter=0)

        Initialize a context with a seed that is used for all computations.

    **Serialization**

    .. cpp:function:: context_t (const boost::property_tree::ptree& json)

        Load a context from a serialized `ptree` structure.

    .. cpp:function:: boost::property_tree::ptree to_ptree() const

        Serialize the context to a Boost `ptree`.

    **Query**

    .. cpp:function:: size_t get_counter()

        Returns the current position in the random stream.

    **Accessors**

    .. cpp:function:: skylark::utility::random_samples_array_t<Distribution> allocate_random_samples_array(size_t size, Distribution& distribution)

        Returns a container of samples drawn from a distribution to be accessed
        as an array. The container contains `size` samples drawn from the
        specified `distribution`.

    .. cpp:function:: std::vector<typename Distribution::result_type> generate_random_samples_array(size_t size, Distribution& distribution)

        Returns a vector of samples drawn from a distribution. The vector
        contains `size` samples drawn from the specified `distribution`.

    .. cpp:function:: skylark::utility::random_array_t allocate_random_array(size_t size)

        Returns a container of random numbers to be accessed as an array. The
        container lazily provides `size` samples.

    .. cpp:function:: int random_int()

        Returns an integer random number.

Local sparse matrices
---------------------

To be added...

Cross matrix-type GEMM and other linear algebra routines
--------------------------------------------------------

To be added...
