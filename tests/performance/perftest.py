from unittest  import TestCase
from decorator import decorator
import pprint

@decorator
def perftest(f, *args, **kwargs):
    """
    Example useage:

    class CWTTest(TestCase):

        @perftest
        def test_sketch_apply(self):
            # ...

    """

    sout = codecs.getwriter('iso-8859-1')(sys.stdout)
    print "\n"
    print "TESTING: %s()" % getattr(f, "__name__", "<unnamed>")
    print "-------------------------------------------------------------------"
    print "\n"
    t_start  = time.time()
    test_out = f(*args, **kwargs)    # run the test
    t_end    = time.time()
    dt       = str((t_end - t_start)*1.00)
    dt_str   = dt[:(dt.find(".") + 4)]
    #TODO: find average/min/max
    print "-------------------------------------------------------------------"
    print 'RESULTS:'
    pprint (out)
    print 'Test finished in %ss' % dt_str
    print "==================================================================="
