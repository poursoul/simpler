/**
 * @file spin_hint.h
 * @brief Platform-specific spin-wait hint for AICPU (real hardware)
 *
 * On real Ascend hardware, AICPU runs on dedicated ARM A55 cores.
 * The ARM yield instruction hints the processor that the current thread
 * is in a spin-wait loop, reducing cache pressure on other cores sharing L2.
 */

#ifndef PLATFORM_A2A3_AICPU_SPIN_HINT_H_
#define PLATFORM_A2A3_AICPU_SPIN_HINT_H_

#define SPIN_WAIT_HINT() ((void)0)

#endif  // PLATFORM_A2A3_AICPU_SPIN_HINT_H_
