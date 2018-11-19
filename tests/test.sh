cd lib
sigm17rudev="tests/data/russian-dev"
sigm17rurtl="tests/data/russian-train-low"
results="tests/data/SIGM17_RU_LOW"
python run_transducer.py --dynet-seed 1 --dynet-mem 500 --dynet-autobatch 0  --transducer=haem --sigm2017format \
--input=100 --feat-input=20 --action-input=100 --pos-emb  --enc-hidden=200 --dec-hidden=200 --enc-layers=1 \
--dec-layers=1   --mlp=0 --nonlin=ReLU --il-optimal-oracle --il-loss=nll --il-beta=0.5 --il-global-rollout \
--dropout=0 --optimization=ADADELTA --l2=0  --batch-size=1 --decbatch-size=25  --patience=15 --epochs=50 \
--tag-wraps=both --param-tying  --mode=il   --beam-width=0 --beam-widths=4  $sigm17rurtl  $sigm17rudev  $results
