#ifndef COLORS_HPP
#define COLORS_HPP

#include <string>

class c {
public:
    static std::string title_color() {
        return should_be_colorful ? "\033[35m" : "";
    }
    
    static std::string label_color() {
        return should_be_colorful ? "\033[36m" : "";
    }

    static std::string value_color() {
        return should_be_colorful ? "\033[33m" : "";
    }
    
    static std::string reset_color() {
        return should_be_colorful ? "\033[0m" : "";
    }

    static std::string error_color() {
        return should_be_colorful ? "\033[31m" : "";
    }

    static std::string success_color() {
        return should_be_colorful ? "\033[32m" : "";
    }

    static std::string line_up() {
        return should_be_colorful ? "\033[1A" : "";
    }

    static std::string extra_line_in_params() {
        return should_be_colorful ? "\n" : "";
    }

    // TIME REPORT COLORS

    static std::string time_report_title() {
        return should_be_colorful ? "\033[1;34m" : "";
    }

    static std::string time_report_labels() {
        return should_be_colorful ? "\033[1;33m" : "";
    }

    static std::string time_report_sublabels() {
        return should_be_colorful ? "\033[33m" : "";
    }

    static std::string time_report_time() {
        return should_be_colorful ? "\033[32m" : "";
    }

    static std::string time_report_positive() {
        return should_be_colorful ? "\033[32m" : "";
    }

    static std::string time_report_negative() {
        return should_be_colorful ? "\033[31m" : "";
    }

    static std::string time_report_info() {
        return should_be_colorful ? "\033[36m" : "";
    }

    // GRID COLORS

    static std::string grid_print_zero() {
        return should_be_colorful ? "\033[30m" : "";
    }

    static std::string grid_print_one() {
        return should_be_colorful ? "\033[31m" : "";
    }

    static void set_colorful(bool colorful) {
        c::should_be_colorful = colorful;
    }

public:
    static bool should_be_colorful;
};

#endif // COLORS_HPP