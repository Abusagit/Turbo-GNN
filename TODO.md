add edditional convolution to dgl backend -- now there is only graphconv

add additional normalizations (right & left) for torch native - to emulate graphconv with left/right/both (gcn) normalizations (I already added required parameters)

add autotune for datasets not only for random graphs in scripts autotune


add val every K steps/epochs

use Enums for Registry instead of strings

