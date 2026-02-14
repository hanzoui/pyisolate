.. pyisolate documentation master file

pyisolate Documentation
=======================

**pyisolate** is a Python library for running extensions across multiple isolated virtual environments with RPC communication.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   rpc_protocol
   debugging
   edge_cases
   platform_compatibility

Overview
--------

pyisolate solves dependency conflicts by isolating extensions in separate venvs while maintaining seamless host-extension communication through AsyncRPC.

Key Features
~~~~~~~~~~~~

* **Dependency Isolation**: Each extension gets its own virtual environment
* **Transparent RPC**: Seamless communication between host and extensions
* **PyTorch Sharing**: Optionally share PyTorch models across processes for memory efficiency
* **Simple API**: Easy to use with minimal configuration

Quick Example
~~~~~~~~~~~~~

.. code-block:: python

   from pyisolate import ExtensionManager

   async def main():
       manager = ExtensionManager("./extensions")
       await manager.start()

       # Extensions can be called transparently
       result = await manager.extensions['my_extension'].process_data(data)

       await manager.stop()

Installation
------------

.. code-block:: bash

   pip install pyisolate

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
