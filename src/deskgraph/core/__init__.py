"""Pure quant library: instruments, market inputs, pricers and greeks.

Nothing in this package does I/O, touches the network, reads the clock or keeps
mutable module-level state. Randomness comes only from a seeded
``numpy.random.Generator`` passed in by the caller. ``tests/test_core_purity.py``
enforces these rules on every module here.
"""
