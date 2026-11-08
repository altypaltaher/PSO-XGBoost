cd "$(dirname "$0")/.."
python3 src/train_v2.py andmal app main 0 full
python3 src/train_v2.py andmal flow main 0 small
python3 src/train_v2.py andmal app ports 0 small
python3 src/train_v2.py andmal app noinitwin 0 small
python3 src/train_v2.py andmal app main 1 full
python3 src/train_v2.py andmal app main 2 full
python3 src/train_v2.py aagm flow main 0 full
echo QUEUE-COMPLETE
