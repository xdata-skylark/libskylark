#ifndef SKYLARK_HDFS_HPP
#define SKYLARK_HDFS_HPP

// Some helper classes to handle HDFS
#if SKYLARK_HAVE_LIBHDFS

#include "hdfs.h"

namespace skylark { namespace utility {

struct hdfs_line_streamer_t {

    hdfs_line_streamer_t(const hdfsFS &fs, const std::string& file_name, int bufsize) :
        _bufsize(bufsize), _fs(fs), _fid(nullptr),
        _readbuf(new char[bufsize]), _eof(false), _closed(false),
        _emptybuf(true), _readsize(0), _loc(0) {

        open(file_name);

    }

    ~hdfs_line_streamer_t() {
        close();
        delete _readbuf;
    }

    //FIXME: Can lines end in the next file?
    void getline(std::string &line) {
        line = "";
        while (!_eof) {
            if (_emptybuf) {
                _readsize = hdfsRead(_fs, _fid, _readbuf, _bufsize - 1);
                if (_readsize == 0) {
                    _eof = true;
                    break;
                }
                _readbuf[_readsize] = 0;
                _emptybuf = false;
                _loc = 0;
            }

            char *el = std::strchr(_readbuf + _loc, '\n');
            if (el != NULL) {
                *el = 0;
                line += std::string(_readbuf + _loc);
                _loc += el - (_readbuf + _loc) + 1;
                if (_loc == _readsize)
                    _emptybuf = true;
                return;
            } else {
                line += std::string(_readbuf + _loc);
                _emptybuf = true;
            }
        }
    }

    //XXX: currently rewind is not used anymore
    void rewind() {
        _eof = false;
        _emptybuf = true;
        _readsize = 0;
        hdfsSeek(_fs, _fid, 0);
    }

    void close() {
        if (!_closed)
            hdfsCloseFile(_fs, _fid);
        _fid = nullptr;
        _closed = true;
    }


    bool eof() {
        return _eof;
    }

private:
    const int _bufsize;
    const hdfsFS &_fs;
    hdfsFile _fid;
    char *_readbuf;
    bool _eof, _closed, _emptybuf;
    int _readsize;
    int _loc;

    void open(const std::string& file_name) {

        close();
        _fid = hdfsOpenFile(_fs, file_name.c_str(), O_RDONLY, 0, 0, 0);

        if(!_fid) {
            std::stringstream ss;
            ss << "Failed to open HDFS file " << file_name;
            SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
                skylark::base::error_msg(ss.str()))
        }
    }
};

/**
 * Helper class to parse a iterate hdfs_line_streamer structures.
 * The helper determines if we have to deal with a single file or a directory.
 */
struct hdfs_line_streamer_iterator_t {

    hdfs_line_streamer_iterator_t(const hdfsFS& fs, const std::string& path,
        int line_stream_bufsize)
        : _fs(fs)
        , _line_stream_bufsize(line_stream_bufsize)
        , _file_idx(0) {

        get_files(path);
    }

    ~hdfs_line_streamer_iterator_t()
    {}

    /**
     * Reset the iterator.
     */
    void reset() {
        _file_idx = 0;
    }

    /**
     * Get the next file_streamer for the next file.
     * Returns a null pointer if the iterator has reached the end of the list.
     */
    std::shared_ptr<hdfs_line_streamer_t> next() {

        if(_file_idx == _file_names.size())
            return nullptr;

        std::string file_name = _file_names[_file_idx];
        _file_idx++;

        return std::shared_ptr<hdfs_line_streamer_t>(new
                hdfs_line_streamer_t(_fs, file_name, _line_stream_bufsize));
    }

private:

    const hdfsFS& _fs;
    int _line_stream_bufsize;
    size_t _file_idx;

    std::vector<std::string> _file_names;

    /**
     * Gathers all file/s given a path that can either be a file name or a
     * directory name.
     * We skip all files that have a size of zero bytes.
     *
     * \param path name of the hdfs path.
     */
    void get_files(const std::string& path) {

        hdfsFileInfo* dir_info = hdfsGetPathInfo(_fs, path.c_str());

        if(dir_info == NULL) {
            std::stringstream ss;
            ss << "Failed to get path infor for " << path;
            SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
                    skylark::base::error_msg(ss.str()))
        }

        if(dir_info->mKind == kObjectKindDirectory) {

            int num_entries = 0;
            hdfsFileInfo* info =
                hdfsListDirectory(_fs, path.c_str(), &num_entries);

            for(int i = 0; i < num_entries; i++) {
                if(info[i].mSize == 0)
                    continue;

                std::string file_name(info[i].mName);

                size_t pos = file_name.find(path);
                std::stringstream ss;
                ss << file_name.substr(pos);
                _file_names.push_back(ss.str());
            }

            hdfsFreeFileInfo(info, num_entries);

        } else { //kObjectKindFile

            _file_names.push_back(path);
        }

        hdfsFreeFileInfo(dir_info, 1);
    }
};

} }

#endif
#endif
