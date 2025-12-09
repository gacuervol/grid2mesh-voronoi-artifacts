# compute_date_splits.py
import argparse
from datetime import datetime
import sys

def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Compute training/validation/test splits from year range."
        )
    parser.add_argument(
        "-s", "--start_date", required=True, help="Start date in YYYY-MM-DD format"
        )
    parser.add_argument(
        "-e", "--end_date", required=True, help="End date in YYYY-MM-DD format"
        )
    
    return parser.parse_args()

def main():

    args = parse_args()
    y0 = datetime.fromisoformat(args.start_date).year
    y1 = datetime.fromisoformat(args.end_date).year
    years = list(range(y0, y1+1))
    n = len(years)

    if n >= 10:
        # 80/10/10 en años
        n_train = int(n * 0.8)
        n_val  = int(n * 0.1)
        # asegúrate de sumar al final si sobra un año
        n_test = n - n_train - n_val
    else:
        # caso pocos años: reserva el penúltimo para val y el último para test
        n_train = n - 2
        n_val   = 1
        n_test  = 1

    train = years[:n_train]
    val   = years[n_train:n_train+n_val]
    test  = years[n_train+n_val:]

    print(f"{train[0]}-01-01 {train[-1]}-12-31 "
          f"{val[0]}-01-01 {val[-1]}-12-31 "
          f"{test[0]}-01-01 {test[-1]}-12-31")

if __name__ == "__main__":
    main()
