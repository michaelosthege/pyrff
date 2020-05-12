from . exceptions import DtypeError, ShapeError
from . rff import sample_rff, save_rffs, load_rffs
from . thompson import sample_batch, get_probabilities
from . utils import multi_start_fmin

__version__ = '0.3.1'
