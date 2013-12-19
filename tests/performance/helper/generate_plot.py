# A simple helper script to generate plots for performance tests

import matplotlib
# do not use any X backend
matplotlib.use('Agg')
import pylab


import csv
def generate_plot_from_file(fname, web_dir):
    print("generating plot for %s" % (fname))
    name = fname[:fname.rfind('_')]

    procs = []
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        import operator
        sorted_data = sorted(reader, key=operator.itemgetter(0))
        ofile = open('tmp.dat', "wb")
        writer = csv.writer(ofile, delimiter=' ')
        for row in sorted_data:
            procs.append(int(row[0]))
            writer.writerow(row)
        ofile.close()

    procs = list(set(procs))

    pylab.plotfile('tmp.dat', cols=(0,1,2,3), delimiter=' ',
            names=['$n_p$', 'min', 'avg', 'max'], subplots=False,
            plotfuncs={1:'semilogy', 2:'semilogy', 3:'semilogy'},
            marker='s')
    pylab.title("%s Performance" % (name))
    pylab.ylabel("time [s]")
    pylab.xticks(procs)
    pylab.savefig("%s/plots/%s.png" % (web_dir, fname))
    return "%s.png" % (fname)

def update_overview(web_dir):
    pylab.ylabel("time [s]")
    #pylab.xticks(date)
    pylab.savefig("%s/overview.png" % (web_dir))
    pass


import os
import glob
from datetime import date
def generate_plots(input_dir, web_dir):
    html = "<h1>Overview</h1>\n"
    update_overview(web_dir)
    html += "<img src=\"overview.png\" width=\"450px\">"

    #FIXME: collect results from other machines...
    html += "<h1>Performance on %s</h1>\n" % (date.today())
    os.chdir(input_dir)
    for infile in glob.glob("*_test_*.perf"):
        png_file = generate_plot_from_file(infile, web_dir)
        html += "<a href=\"plots/%s\">" % (png_file)
        html += "<img src=\"plots/%s\" width=\"450px\">" % (png_file)
        html += "</a>\n"

    html += "<hr>\n"

    html += "<h1>Archive</h1>\n"

    with open(web_dir + "/latest.html", "w") as out:
        out.write(html)


import sys
if __name__ == "__main__":
    generate_plots(sys.argv[1], sys.argv[2])
