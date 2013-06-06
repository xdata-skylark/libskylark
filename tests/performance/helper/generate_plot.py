# A simple helper script to create plots for performance tests

import matplotlib
# do not use any X backend
matplotlib.use('Agg')
import pylab
import csv

from datetime import date


def generate_plot(fname, output_dir):
    print("generating plot for %s" % (fname))
    name = fname[:fname.rfind('_')]

    procs = []
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            procs.append(int(row[0]))

    procs = list(set(procs))

    pylab.plotfile(fname, cols=(0,1,2,3), delimiter=' ',
            names=['$n_p$', 'min', 'avg', 'max'], subplots=False,
            plotfuncs={1:'semilogy', 2:'semilogy', 3:'semilogy'})
    pylab.title("%s Performance" % (name))
    pylab.ylabel("time [s]")
    pylab.xticks(procs)
    pylab.savefig("%s/%s.png" % (output_dir, fname))
    return "%s.png" % (fname)


import os
import glob
def generate_plots(input_dir, output_dir):
    html = "<h1>Performance on %s</h1>\n" % (date.today())
    os.chdir(input_dir)
    for infile in glob.glob("*_test_*.perf"):
        png_file = generate_plot(infile, output_dir)
        html += "<a href=\"%s\">\n" % (png_file)
        html += "<img src=\"%s\" width=\"450px\">\n" % (png_file)
        html += "</a>\n"

    html += "<hr>\n"

    with open(output_dir + "/latest.html", "w") as out:
        out.write(html)


import sys
if __name__ == "__main__":
    generate_plots(sys.argv[1], sys.argv[2])
