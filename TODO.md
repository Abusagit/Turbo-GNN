add edditional convolution to dgl backend -- now there is only graphconv

add additional normalizations (right & left) for torch native - to emulate graphconv with left/right/both (gcn) normalizations (I already added required parameters)

add autotune for datasets not only for random graphs in scripts autotune


cache all graph representations for every backend -- remove the logic from forwards

add unified tqdm for all epochs + add val every K steps/epochs


use Enums for Registry instead of strings

