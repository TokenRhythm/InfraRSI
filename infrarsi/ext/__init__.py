"""External evaluation environments (EXT_INTERFACES.md): shared layer `common/` plus one adapter per environment
(`rebench_llmf/` ENV-A, `optico/` ENV-B, `autoresearch/` ENV-C). The environment's own judge is the score authority;
InfraRSI guards only add rejections."""
