QT += core
QT -= gui

TARGET = CSK_repo
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    ../src/benchmark_info.cpp \
    ../src/csk.cpp \
    ../src/main.cpp \
    ../src/run_tracker.cpp

HEADERS += \
    ../src/benchmark_info.h \
    ../src/csk.h

