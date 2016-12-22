import argparse
from model.Evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--recom", nargs="+", type=str, help="the exponent", default='all')
parser.add_argument("--hidden", nargs="+", type=int, help="dimension of hidden", default=[100])
parser.add_argument("--small", type=bool, help="for small set?", default=False)
parser.add_argument("--batch_size", type=int, help="batch size", default=256)
parser.add_argument("--demo", type=bool, help="run demo?", default=False)

args = parser.parse_args()
evaluator = Evaluator(args.recom, args.hidden, True, args.batch_size)
evaluator.evaluate()
if args.demo:
    evaluator.run_demo()

