#include "MyUtils.h"

std::string myutils::getDatetimeStr() {
    time_t t = time(nullptr);
    errno_t error;
    struct tm localTime;
    error = localtime_s(&localTime, &t);
    std::stringstream s;
    s << "20" << localTime.tm_year - 100;
    // setw(),setfill()‚Å0‹l‚ß
    s << std::setw(2) << std::setfill('0') << localTime.tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime.tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime.tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime.tm_min;
    s << std::setw(2) << std::setfill('0') << localTime.tm_sec;
    // std::string‚É‚µ‚Ä’l‚ð•Ô‚·
    return s.str();
}