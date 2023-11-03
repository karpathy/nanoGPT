"""
ngpt/__init__.py
"""
from __future__ import absolute_import, annotations, division, print_function
# import logging
# import os
# from typing import Optional
# import warnings
# import torch
#
# from mpi4py import MPI
# # from rich.logging import RichHandler
# # from l2hmc.utils.enrich import EnRichHandler
# from enrich.handler import RichHandler
# import tqdm
# from rich import print

# warnings.filterwarnings('ignore')

# os.environ['PYTHONIOENCODING'] = 'utf-8'
#
# RANK = int(MPI.COMM_WORLD.Get_rank())
# WORLD_SIZE = int(MPI.COMM_WORLD.Get_size())
#
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {DEVICE}")

# # Check that MPS is available
# if (
#         torch.backends.mps.is_available()
#         and torch.get_default_dtype() != torch.float64
# ):
#     DEVICE = torch.device("mps")
# elif not torch.backends.mps.is_built():
#     DEVICE = 'cpu'
#     print(
#         "MPS not available because the current PyTorch install was not "
#         "built with MPS enabled."
#     )
# else:
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {DEVICE}")
#
#
# class DummyTqdmFile(object):
#     """ Dummy file-like that will write to tqdm
#     https://github.com/tqdm/tqdm/issues/313
#     """
#     file = None
#
#     def __init__(self, file):
#         self.file = file
#
#     def write(self, x):
#         # Avoid print() second call (useless \n)
#         # if len(x.rstrip()) > 0:
#         tqdm.tqdm.write(x, file=self.file, end='\n')
#
#     def flush(self):
#         return getattr(self.file, "flush", lambda: None)()
#
#
# def get_rich_logger(
#         name: Optional[str] = None,
#         level: str = 'INFO'
# ) -> logging.Logger:
#     # log: logging.Logger = get_logger(name=name, level=level)
#     log = logging.getLogger(name)
#     log.handlers = []
#     from enrich.console import get_console
#     # from l2hmc.utils.rich import get_console
#     console = get_console(
#         markup=True,
#         redirect=(WORLD_SIZE > 1),
#     )
#     handler = RichHandler(
#         level,
#         rich_tracebacks=False,
#         console=console,
#         show_path=False,
#         enable_link_path=False
#     )
#     log.handlers = [handler]
#     log.setLevel(level)
#     return log
#
#
# def get_file_logger(
#         name: Optional[str] = None,
#         level: str = 'INFO',
#         rank_zero_only: bool = True,
#         fname: Optional[str] = None,
#         # rich_stdout: bool = True,
# ) -> logging.Logger:
#     # logging.basicConfig(stream=DummyTqdmFile(sys.stderr))
#     import logging
#     fname = 'ngpt' if fname is None else fname
#     log = logging.getLogger(name)
#     if rank_zero_only:
#         fh = logging.FileHandler(f"{fname}.log")
#         if RANK == 0:
#             log.setLevel(level)
#             fh.setLevel(level)
#         else:
#             log.setLevel('CRITICAL')
#             fh.setLevel('CRITICAL')
#     else:
#         fh = logging.FileHandler(f"{fname}-{RANK}.log")
#         log.setLevel(level)
#         fh.setLevel(level)
#     # create formatter and add it to the handlers
#     formatter = logging.Formatter(
#         "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
#     )
#     fh.setFormatter(formatter)
#     log.addHandler(fh)
#     return log
#
#
# def get_logger(
#         name: Optional[str] = None,
#         level: str = 'INFO',
#         rank_zero_only: bool = True,
#         **kwargs,
# ) -> logging.Logger:
#     log = logging.getLogger(name)
#     # log.handlers = []
#     # from rich.logging import RichHandler
#     # from l2hmc.utils.rich import get_console, is_interactive
#     from enrich.console import get_console, is_interactive
#     # format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
#     if rank_zero_only:
#         if RANK != 0:
#             log.setLevel('CRITICAL')
#         else:
#             log.setLevel(level)
#     if RANK == 0:
#         console = get_console(
#             markup=True,  # (WORLD_SIZE == 1),
#             redirect=(WORLD_SIZE > 1),
#             **kwargs
#         )
#         if console.is_jupyter:
#             console.is_jupyter = False
#         # log.propagate = True
#         # log.handlers = []
#         # use_markup = (
#         #     WORLD_SIZE == 1
#         #     and not is_interactive()
#         # )
#         log.addHandler(
#             RichHandler(
#                 omit_repeated_times=False,
#                 level=level,
#                 console=console,
#                 show_time=True,
#                 show_level=True,
#                 show_path=True,
#                 markup=True,
#                 enable_link_path=(WORLD_SIZE == 1 and not is_interactive()),
#             )
#         )
#         log.setLevel(level)
#     if (
#             len(log.handlers) > 1
#             and all([i == log.handlers[0] for i in log.handlers])
#     ):
#         log.handlers = [log.handlers[0]]
#     return log
