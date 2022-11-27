import os
import argparse
from solver.solver import Solver


parser = argparse.ArgumentParser('Perceptual Reflection Removel')
#train
parser.add_argument('--data_path_single',default='./SDD/train_patches/',help="synthetic data")
parser.add_argument('--kpn',default='kpn',help="kpn or kpn-onebranch")
parser.add_argument('--save_model_freq',default=10,type=int,help="frequency to save model")
parser.add_argument('--test_model_freq',default=1,type=int,help="frequency to test model")
parser.add_argument('--print_freq',type=int,default=1000,help='print frequency (default: 10)')
parser.add_argument('--weight_save_path',default='./checkpoint_sdd',help="save path")

parser.add_argument('--resume_file',default='',help="resume file path")
parser.add_argument('--fine_tune',default='',help="fine tune path")
parser.add_argument('--lr',default=2e-5,type=float,help="learning rate")
parser.add_argument('--lr_decay',default=60,type=int,help="learning rate")
parser.add_argument('--load_workers',default=8,type=int,help="number of workers to load data")
parser.add_argument('--batch_size',default=1,type=int,help="batch size")
parser.add_argument('--start_epoch',type=int,default=0,help="start epoch of training")
parser.add_argument('--num_epochs',type=int,default=100,help="total epoch of training")

#eval
parser.add_argument('--test_path',default='./SDD/test/',help="path to test set")
parser.add_argument('--save_result_path',default=True,help="if save result")
parser.add_argument('--model',default='UNet',help="UNet only")
parser.add_argument('--num_workers',default=4,help="num_workers")

def main():
	if not os.path.exists('./summary'):
		os.mkdir('summary')
	args = parser.parse_args()
	solver=Solver(args) 
	solver.train_model()

if __name__=='__main__':
	main()
