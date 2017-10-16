#ifndef __SYNTHESIS_CODEGEN_H
#define __SYNTHESIS_CODEGEN_H

#include <string>
#include <sstream>
#include <iostream>
#include <cinttypes>
#include <map>
#include <vector>

namespace glang {

class CodeFormatter {
 public:
  CodeFormatter() : indent_(0) {}

  std::string indentation() const {
    std::string res = "";
    for (int i = 0; i < indent_; i++) {
      res += " ";
    }

    return res;
  }

  CodeFormatter &enter() {
    indent_ += 2;
    return *this;
  }

  CodeFormatter &exit() {
    indent_ -= 2;
    return *this;
  }

  CodeFormatter &line(std::string s) {
    code_ += indentation();
    code_ += s;
    code_ += "\n";
    return *this;
  }

  CodeFormatter &open_bracket() {
    return line("{").enter();
  }

  CodeFormatter &close_bracket() {
    return exit().line("}");
  }

 private:
  int indent_;
  std::string code_;
};

}  // namespace glang

#endif  // __SYNTHESIS_CODEGEN_H
