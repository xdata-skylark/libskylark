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
            marker='s', linewidth=3)
    pylab.title("%s Performance" % (name))
    pylab.ylabel("time [s]")
    pylab.xticks(procs)
    pylab.savefig("%s/plots/%s.png" % (web_dir, fname))
    return "%s.png" % (fname)


def compute_efficiency(fname):
    """
    Currently just uses min and max processor results found in file to determine
    the parallel efficiency.
    """
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        import operator
        sorted_data = sorted(reader, key=operator.itemgetter(0))
        r1 = sorted_data[0]
        rk = sorted_data[-1]
        return (float(r1[2]) / float(rk[2])) / (int(rk[0]) / float(r1[0]))


def update_overview(web_dir):
    """
    plot parallel efficiency of last month
    """
    import datetime
    today = datetime.date.today()
    dates = [ str(today - datetime.timedelta(days=x)) for x in range(30, -1, -1) ]

    perf = {}
    for date in dates:
        for infile in glob.glob("*_test_*_%s.perf" % date):
            name = infile[:infile.rfind('_')]
            try:
                perf[name].append((date, compute_efficiency(infile)))
            except KeyError:
                perf[name] = [(date, compute_efficiency(infile))]

    pylab.rcParams['figure.figsize'] = 20, 10
    actual_dates = []
    for name in perf:
        actual_dates, vals = zip(*perf[name])
        pylab.plot(vals, hold=True, label="%s" % name, linewidth=3)

    pylab.xlabel("date")
    pylab.xticks(range(len(actual_dates)), actual_dates, rotation='vertical')
    pylab.yticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    pylab.ylabel("parallel efficiency [%]")
    pylab.grid(True)
    pylab.legend()
    pylab.savefig("%s/overview.png" % (web_dir))
    pass


import os
import glob
from datetime import date
def generate_plots(data_dir, web_dir):
    os.chdir(data_dir)

    html = "<html><head><link href='http://fonts.googleapis.com/css?family=Voces' rel='stylesheet' type='text/css'></head>"
    html += "<body><div style=\"width:960px; display: block; margin-left: auto; margin-right: auto;\">\n"
    html += "<h1 style=\"font-family: 'Voces';\">Overview</h1>\n"
    update_overview(web_dir)
    html += "<img src=\"overview.png\" width=\"960px\">"

    #FIXME: collect results from other machines...
    html += "<h1 style=\"font-family: 'Voces';\">Performance on %s</h1>\n" % (date.today())
    # generate todays plots
    for infile in glob.glob("*_test_*_%s.perf" % date.today()):
        png_file = generate_plot_from_file(infile, web_dir)
        html += "<a href=\"plots/%s\">" % (png_file)
        html += "<img src=\"plots/%s\" width=\"450px\">" % (png_file)
        html += "</a>\n"

    html += "<hr>\n"

    html += "<h1 style=\"font-family: 'Voces';\">Archive</h1>\n"

    archived = []
    for infile in glob.glob("*_test_*.perf"):
        archived.append(infile.split('_')[-1].split(".")[0])

    archived = sorted(set(archived), reverse=True)
    html += "<ul style=\"font-family: 'Voces';\">\n"
    #FIXME: generate pages in advance?
    for ar in archived:
        html += "<li><a href=\"%s.html\">%s</a></li>\n" % (ar, ar)

    html += "</ul>\n"
    html += "</div></body></html>\n"

    with open(web_dir + "/latest.html", "w") as out:
        out.write(html)


import sys
if __name__ == "__main__":
    generate_plots(sys.argv[1], sys.argv[2])
