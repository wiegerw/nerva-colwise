// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/nerva_timer.h
/// \brief add your file description here.

#pragma once

#include "nerva/utilities/timer.h"

inline nerva::utilities::suspendable_timer nerva_timer;

namespace nerva
{

inline void print_nerva_timer_report()
{
#ifndef NERVA_DISABLE_TIMER
  nerva_timer.print_report();
#endif
}

}

#ifndef NERVA_DISABLE_TIMER
#define NERVA_TIMER_START(name) nerva_timer.start(name);
#define NERVA_TIMER_STOP(name) nerva_timer.stop(name);
#else
#define NERVA_TIMER_START(name)
#define NERVA_TIMER_STOP(name)
#endif
