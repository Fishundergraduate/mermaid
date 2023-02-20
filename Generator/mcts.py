import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import hydra
from config.config import cs
from omegaconf import DictConfig
import torch
import time
import warnings
warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle

import torch.nn.functional as F

from Model.model import RolloutNetwork
from Utils.pareto_utils import read_smilesset, parse_smiles, convert_smiles, RootNode, ParentNode, NormalNode, \
    trans_infix_ringnumber, id
from Utils.utils import VOCABULARY
from Utils.reward import getReward, getRewards, DockingReward
from Utils.sascore import calculateScore

import json
# Load Configure
from omegaconf import DictConfig, OmegaConf
from config.config import Config
from hydra.utils import instantiate
# calc for pareto
from pygmo import hypervolume
import math
import copy

# for debug
import pdb
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCTS(object):
    def __init__(self, init_smiles, model, vocab, Reward, max_seq=81, c=1, num_prll=256, limit=5, step=0, n_valid=0,
                 n_invalid=0, sampling_max=False, max_r=-1000):
        self.init_smiles = parse_smiles(init_smiles.rstrip("\n"))
        self.model = model
        self.vocab = vocab
        self.Reward = Reward
        self.max_seq = max_seq
        self.valid_smiles = {}
        self.c = c
        self.count = 0
        self.ub_prll = num_prll
        self.limit = np.sum([len(self.init_smiles)+1-i for i in range(limit)])
        self.sq = set([s for s in self.vocab if "[" in s])
        self.max_score = max_r
        self.step = step
        self.n_valid = n_valid
        self.n_invalid = n_invalid
        self.sampling_max = sampling_max
        self.total_nodes = 0

    def select(self):
        raise NotImplementedError()

    def expand(self):
        raise NotImplementedError()

    def simulate(self):
        raise NotImplementedError()

    def backprop(self):
        raise NotImplementedError()

    def search(self, n_step):
        raise NotImplementedError()


class ParseSelectMCTS(MCTS):
    def __init__(self, *args, **kwargs):
        super(ParseSelectMCTS, self).__init__(*args, **kwargs)
        self.root = RootNode()
        self.current_node = None
        self.next_token = {}
        self.rollout_result = {}
        self.l_replace = int(len(self.init_smiles)/4)

    def select(self):
        """
        search for the node with no child nodes and maximum UCB score
        """
        self.current_node = self.root
        while len(self.current_node.children) != 0:
            self.current_node = self.current_node.select_children()
            if self.current_node.depth+1 > self.max_seq:
                tmp = self.current_node
                # update
                while self.current_node is not None:
                    self.current_node.cum_score += -1
                    self.current_node.visit += 1
                    self.current_node = self.current_node.parent
                tmp.remove_Node()

                self.current_node = self.root

    def expand(self, epsilon=0.1, loop=10, gamma=0.90):
        """

        """
        # Preparation of prediction using RNN model, list -> tensor
        x = np.zeros([1, self.max_seq])
        c_path = convert_smiles(self.current_node.path[2:], self.vocab, mode="s2i")
        x[0, :len(c_path)] = c_path
        x = torch.tensor(x, dtype=torch.long)
        x_len = [len(c_path)]

        # Predict the probabilities of next token following current node
        with torch.no_grad():
            y = self.model(x, x_len)
            y = F.softmax(y, dim=2)
            y = y.to('cpu').detach().numpy().copy()
            y = np.array(y[0, len(self.current_node.path)-3, :])
            y = np.log(y)
            prob = np.exp(y) / np.sum(np.exp(y))

        # Sampling next token based on the probabilities
        self.next_token = {}
        while len(self.next_token) == 0:
            for j in range(loop):
                if np.random.rand() > epsilon * (gamma ** len(self.current_node.path)):
                    ind = np.random.choice(range(len(prob)), p=prob)
                else:
                    ind = np.random.randint(len(self.vocab))
                self.next_token[self.vocab[ind]] = 0
            if self.current_node.depth == 1:
                self.next_token["("] = 0
        self.check()

        print("".join(self.current_node.path[2:]), len(self.next_token))
        print(self.next_token.keys())

    def check(self):
        if "\n" in self.next_token.keys():
            tmp_node = self.current_node
            while tmp_node.depth != 1:
                tmp_node = tmp_node.parent
            original_smiles = tmp_node.original_smiles
            pref, suf = original_smiles.split("*")
            inf = "".join(self.current_node.path[3:])
            smiles_concat = pref + trans_infix_ringnumber(pref, inf) + suf

            score = self.Reward.reward(smiles_concat)

            self.max_score = max(self.max_score, score)
            self.next_token.pop("\n")
            if score > -100:
                self.valid_smiles["%d:%s" % (-self.step, smiles_concat)] = score
                print(score, smiles_concat)
                self.max_score = max(self.max_score, score)
                self.n_valid += 1# TODO: 1 ligands and scores outputter
            else:
                self.n_invalid += 1

        if len(self.next_token) < 1:
            self.current_node.cum_score = -100000
            self.current_node.visit = 100000
            self.current_node.remove_Node()

    def simulate(self):
        tmp_node = self.current_node
        while tmp_node.depth != 1:
            tmp_node = tmp_node.parent
        original_smiles = tmp_node.original_smiles
        pref, suf = original_smiles.split("*")
        self.rollout_result = {}

        #######################################

        l = len(self.current_node.path)
        part_smiles = [[] for i in range(len(self.next_token))]
        x = np.zeros([len(self.next_token), self.max_seq])
        x_len = []
        for i, k in enumerate(self.next_token.keys()):
            part_smiles[i].extend(self.current_node.path[2:])
            part_smiles[i].append(k)
            x[i, :len(part_smiles[i])] = convert_smiles(part_smiles[i], self.vocab, mode="s2i")
            x_len.append(len(part_smiles[i]))
        x = torch.tensor(x, dtype=torch.long)

        is_terminator = [True]*len(self.next_token)
        step = 0

        while np.sum(is_terminator) > 0 and step+l < self.max_seq-1:
            with torch.no_grad():
                y = self.model(x, x_len)
                y = F.softmax(y, dim=2)
                y = y.to('cpu').detach().numpy().copy()
                prob = y[:, step+l-2, :]

            if self.sampling_max:
                ind = np.argmax(prob, axis=1)
            else:
                ind = [np.random.choice(range(len(self.vocab)), p=prob[i]) for i in range(len(self.next_token))]

            for i in range(len(x_len)):
                x_len[i] += 1

            for i in range(len(self.next_token)):
                x[i, step+l-1] = ind[i]
                if is_terminator[i] and ind[i] == self.vocab.index("\n"):
                    is_terminator[i] = False
                    inf = "".join(convert_smiles(x[i, 1:step+l-1], self.vocab, mode="i2s"))
                    smiles_concat = pref + trans_infix_ringnumber(pref, inf) + suf

                    score = self.Reward.reward(smiles_concat)

                    self.next_token[list(self.next_token.keys())[i]] = score
                    self.rollout_result[list(self.next_token.keys())[i]] = (smiles_concat, score)
                    if score > self.Reward.vmin:
                        # self.valid_smiles[smiles_concat] = score
                        self.valid_smiles["%d:%s" % (self.step, smiles_concat)] = score
                        self.max_score = max(self.max_score, score)
                        print(score, smiles_concat)
                        self.n_valid += 1# TODO: 1 ligands and scores outputter
                    else:
                        # print("NO", smiles_concat)
                        self.n_invalid += 1
            step += 1

    def backprop(self):
        for i, key in enumerate(self.next_token.keys()):
            child = NormalNode(key, c=self.c)
            child.id = self.total_nodes
            self.total_nodes += 1
            try:
                child.rollout_result = self.rollout_result[key]
            except KeyError:
                child.rollout_result = ("Termination", [-10000 for r in self.Reward])
            self.current_node.add_Node(child)
        max_reward = max(self.next_token.values())
        # self.max_score = max(self.max_score, max_reward)
        while self.current_node is not None:
            self.current_node.visit += 1
            self.current_node.cum_score += max_reward/(1+abs(max_reward))
            self.current_node.imm_score = max(self.current_node.imm_score, max_reward/(1+abs(max_reward)))
            self.current_node = self.current_node.parent

    def search(self, n_step, epsilon=0.1, loop=10, gamma=0.90, rep_file=None):
        self.set_repnode(rep_file=rep_file)

        while self.step < n_step:
            self.step += 1
            print("--- step %d ---" % self.step)
            print("MAX_SCORE:", self.max_score)
            if self.n_valid+self.n_invalid == 0:
                valid_rate = 0
            else:
                valid_rate = self.n_valid/(self.n_valid+self.n_invalid)
            print("Validity rate:", valid_rate)
            self.select()
            self.expand(epsilon=epsilon, loop=loop, gamma=gamma)
            if len(self.next_token) != 0:
                self.simulate()
                self.backprop()

    def set_repnode(self, rep_file=None):
        if len(rep_file) > 0:
            for smiles in read_smilesset(hydra.utils.get_original_cwd()+rep_file):
                n = ParentNode(smiles)
                self.root.add_Node(n)
                c = NormalNode("&")
                n.add_Node(c)
        else:
            for i in range(self.l_replace+1):
                for j in range(len(self.init_smiles)-i+1):
                    infix = self.init_smiles[j:j+i]
                    prefix = "".join(self.init_smiles[:j])
                    suffix = "".join(self.init_smiles[j + i:])

                    sc = prefix + "(*)" + suffix
                    mol_sc = Chem.MolFromSmiles(sc)
                    if mol_sc is not None:
                        n = ParentNode(prefix + "(*)" + suffix)
                        self.root.add_Node(n)
                        c = NormalNode("&")
                        n.add_Node(c)

    def save_tree(self, dir_path):
        for i in range(len(self.root.children)):
            stack = []
            stack.extend(self.root.children[i].children)
            sc = self.root.children[i].original_smiles
            score = [self.root.children[i].cum_score]
            ids = [-1]
            parent_id = [-1]
            children_id = [[c.id for c in self.root.children[i].children]]
            infix = [sc]
            rollout_smiles = ["Scaffold"]
            rollout_score = [-10000]

            while len(stack) > 0:
                c = stack.pop(-1)
                for gc in c.children:
                    stack.append(gc)

                # save information
                score.append(c.cum_score)
                ids.append(c.id)
                parent_id.append(c.parent.id)
                ch_id = [str(gc.id) for gc in c.children]
                children_id.append(",".join(ch_id))
                infix.append("".join(c.path))
                rollout_smiles.append(c.rollout_result[0])
                rollout_score.append(c.rollout_result[1])

            df = pd.DataFrame(columns=["ID", "Score", "P_ID", "C_ID", "Infix", "Rollout_SMILES", "Rollout_Score"])
            df["ID"] = ids
            df["Score"] = score
            df["P_ID"] = parent_id
            df["C_ID"] = children_id
            df["Infix"] = infix
            df["Rollout_SMILES"] = rollout_smiles
            df["Rollout_Score"] = rollout_score

            df.to_csv(dir_path+f"/tree{i}.csv", index=False)

class ParseParetoSelectMCTS(MCTS):
    """Pareto Multi-objective optimization Monte Carlo Tree Search class
    """
    def __init__(self, init_smiles, model, vocab, Reward, Pareto, Config: DictConfig, max_seq=81, c=1, num_prll=256, limit=5, step=0, n_valid=0,
                 n_invalid=0, sampling_max=False, max_r=-1000):
        super(ParseParetoSelectMCTS, self).__init__(init_smiles, model, vocab, Reward, max_seq, c, num_prll, limit, step, n_valid,
                 n_invalid, sampling_max, max_r)
        self.current_node = None
        self.next_token = {}
        self.rollout_result = {}
        self.l_replace = int(len(self.init_smiles)/4)
        self.pareto = Pareto
        self.Config = Config
        self.root = RootNode(c=1/np.sqrt(2), cfg = instantiate(Config))
        self.sascore_threshold = Config["mcts"]["sascore_threshold"]

    def select(self):
        """
        search for the node with no child nodes and maximum UCB score
        """
        self.current_node = self.root
        while len(self.current_node.children) != 0:
            self.current_node = self.current_node.select_children(self.pareto, self.Config)
            if self.current_node.depth+1 > self.max_seq:
                tmp = self.current_node
                # update
                while self.current_node is not None:
                    self.current_node.cum_score = list(np.array(self.current_node.cum_score) + np.array([1 for i in range(self.current_node.dimension) ]))
                    self.current_node.visit += 1
                    self.current_node = self.current_node.parent
                tmp.remove_Node()

                self.current_node = self.root

    def expand(self, epsilon=0.1, loop=10, gamma=0.90):
        """

        """
        # Preparation of prediction using RNN model, list -> tensor
        x = np.zeros([1, self.max_seq])
        c_path = convert_smiles(self.current_node.path[2:], self.vocab, mode="s2i")
        x[0, :len(c_path)] = c_path
        x = torch.tensor(x, dtype=torch.long)
        x_len = [len(c_path)]

        # Predict the probabilities of next token following current node
        with torch.no_grad():
            y = self.model(x, x_len)
            y = F.softmax(y, dim=2)
            y = y.to('cpu').detach().numpy().copy()
            y = np.array(y[0, len(self.current_node.path)-3, :])
            y = np.log(y)
            prob = np.exp(y) / np.sum(np.exp(y))

        # Sampling next token based on the probabilities
        self.next_token = {}
        while len(self.next_token) == 0:
            for j in range(loop):
                if np.random.rand() > epsilon * (gamma ** len(self.current_node.path)):
                    ind = np.random.choice(range(len(prob)), p=prob)
                else:
                    ind = np.random.randint(len(self.vocab))
                self.next_token[self.vocab[ind]] = 0
            if self.current_node.depth == 1:
                self.next_token["("] = 0
        self.check()

        print("".join(self.current_node.path[2:]), len(self.next_token))
        print(self.next_token.keys())

    def check(self):
        if "\n" in self.next_token.keys():
            tmp_node = self.current_node
            while tmp_node.depth != 1:
                tmp_node = tmp_node.parent
            original_smiles = tmp_node.original_smiles
            pref, suf = original_smiles.split("*")
            inf = "".join(self.current_node.path[3:])
            smiles_concat = pref + trans_infix_ringnumber(pref, inf) + suf
            mol = Chem.MolFromSmiles(smiles_concat)
            if mol is None:
                #delKeyList.append(list(self.next_token.keys())[i])
                #continue
                self.next_token[list(self.next_token.keys())[i]] = [-1 for r in range(len(self.Reward))]
                return #TODO: check this code to delKey
            elif not isinstance(mol, Chem.rdchem.Mol):
                self.next_token[list(self.next_token.keys())[i]] = [-1 for r in range(len(self.Reward))]
                return
            else:
                smiles_concat = Chem.MolToSmiles(mol)
            sascore = calculateScore(mol)
            if sascore <= self.sascore_threshold:
                scores = []
                for reward in self.Reward:
                    scores.append(reward.reward(smiles_concat))
                score = scores[0]# for compatibility: TODO: Delete this lin
                flag = True
                #TODO: Write by lambda expr.
                for i, score in enumerate(scores):
                    if score <= self.Reward[i].vmin:
                        flag= False
                        break
                
                if flag:
                    self.valid_smiles[smiles_concat] = scores
                    self.valid_smiles["%d:%s" % (self.step, smiles_concat)] = scores
                    #self.max_score = max(self.max_score, score)
                    if self.pareto.dominated(scores):
                        self.pareto.update(scores, smiles_concat, hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"])
                    print(scores, smiles_concat)
                    self.logging(scores=scores,compound=smiles_concat)
                    self.n_valid += 1
                else:
                    # print("NO", smiles_concat)
                    self.next_token[list(self.next_token.keys())[i]] = [-1 for r in range(len(self.Reward))]
                    self.n_invalid += 1

        if len(self.next_token) < 1:
            self.current_node.cum_score = -100000
            self.current_node.visit = 100000
            self.current_node.remove_Node()

    def simulate(self):
        tmp_node = self.current_node
        while tmp_node.depth != 1:
            tmp_node = tmp_node.parent
        original_smiles = tmp_node.original_smiles
        pref, suf = original_smiles.split("*")
        self.rollout_result = {}

        #######################################

        l = len(self.current_node.path)
        part_smiles = [[] for i in range(len(self.next_token))]
        x = np.zeros([len(self.next_token), self.max_seq])
        x_len = []
        for i, k in enumerate(self.next_token.keys()):
            part_smiles[i].extend(self.current_node.path[2:])
            part_smiles[i].append(k)
            x[i, :len(part_smiles[i])] = convert_smiles(part_smiles[i], self.vocab, mode="s2i")
            x_len.append(len(part_smiles[i]))
        x = torch.tensor(x, dtype=torch.long)

        is_terminator = [True]*len(self.next_token)
        step = 0

        while np.sum(is_terminator) > 0 and step+l < self.max_seq-1:
            with torch.no_grad():
                y = self.model(x, x_len)
                y = F.softmax(y, dim=2)
                y = y.to('cpu').detach().numpy().copy()
                prob = y[:, step+l-2, :]

            if self.sampling_max:
                ind = np.argmax(prob, axis=1)
            else:
                ind = [np.random.choice(range(len(self.vocab)), p=prob[i]) for i in range(len(self.next_token))]

            for i in range(len(x_len)):
                x_len[i] += 1
            delKeyList = []
            for i in range(len(self.next_token)):
                x[i, step+l-1] = ind[i]
                if is_terminator[i] and ind[i] == self.vocab.index("\n"):
                    is_terminator[i] = False
                    inf = "".join(convert_smiles(x[i, 1:step+l-1], self.vocab, mode="i2s"))
                    smiles_concat = pref + trans_infix_ringnumber(pref, inf) + suf
                    mol = Chem.MolFromSmiles(smiles_concat)
                    if mol is None:
                        delKeyList.append(list(self.next_token.keys())[i])
                        self.next_token[list(self.next_token.keys())[i]] = [-1 for r in range(len(self.Reward))]
                        continue
                    else:
                        smiles_concat = Chem.MolToSmiles(mol)
                    if not isinstance(smiles_concat, str):
                        self.next_token[list(self.next_token.keys())[i]] = [-1 for r in range(len(self.Reward))]
                        continue
                    #score = self.Reward.reward(smiles_concat)
                    try:
                        sascore = calculateScore(mol)
                    except Exception as e:
                        import traceback
                        print(mol)
                        traceback.print_exc()

                    #Ring Penalty bigger 7 atom
                    ssr = Chem.GetSymmSSSR(mol)
                    if np.any(np.array(list(map(len, ssr)))>=7):
                        self.next_token[list(self.next_token.keys())[i]] = [-1 for r in range(len(self.Reward))]
                        continue
                    if sascore > self.sascore_threshold:
                        self.next_token[list(self.next_token.keys())[i]] = [-1 for r in range(len(self.Reward))]
                        continue
                    scores = []
                    for reward in self.Reward:
                        scores.append(reward.reward(smiles_concat))
                    score = scores[0]# for compatibility: TODO: Delete this line

                    self.next_token[list(self.next_token.keys())[i]] = scores
                    self.rollout_result[list(self.next_token.keys())[i]] = (smiles_concat, scores)
                    flag = True
                    #TODO: Write by lambda expr.
                    for i, score in enumerate(scores):
                        if score <= self.Reward[i].vmin:
                            flag= False
                            break
                    
                    if flag:
                        self.valid_smiles[smiles_concat] = scores
                        self.valid_smiles["%d:%s" % (self.step, smiles_concat)] = scores
                        #self.max_score = max(self.max_score, score)
                        if self.pareto.dominated(scores):
                            self.pareto.update(scores, smiles_concat, hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"])
                        print(scores, smiles_concat)
                        self.logging(scores=scores,compound=smiles_concat)
                        self.n_valid += 1
                    else:
                        # print("NO", smiles_concat)
                        self.next_token[list(self.next_token.keys())[i]] = [-1 for r in range(len(self.Reward))]
                        self.n_invalid += 1
            for key in delKeyList:
                self.next_token.pop(key)
            step += 1

    def backprop(self):
        for i, key in enumerate(self.next_token.keys()):
            child = NormalNode(key, c=self.c, cfg=self.Config)
            child.id = self.total_nodes
            self.total_nodes += 1
            try:
                child.rollout_result = self.rollout_result[key]
            except KeyError:
                child.rollout_result = ("Termination", -10000)
            self.current_node.add_Node(child)
        delKeyList = []
        for key, value in self.next_token.items():
            if not isinstance(value, list):
                #print(f"Line480: ScoreError\t{key, value}")
                delKeyList.append(key)
        for key in delKeyList:
            value = self.next_token.pop(key)
            #print(f"L524: del key: {key}\t value: {value}")
        #print(f"[pdb]:{self.next_token.keys()}\t values:{self.next_token.values()}")
        if len(self.next_token) < 1:
            return
        max_reward = list(np.mean(np.array(list(self.next_token.values())), axis=0)) # Changed to mean from max
        # self.max_score = max(self.max_score, max_reward)
        while self.current_node is not None:
            self.current_node.visit += 1
            #self.current_node.cum_score += max_reward/(1+abs(max_reward))
            self.current_node.cum_score  = list(np.array(self.current_node.cum_score) + np.divide(np.array(max_reward), np.abs(np.array(max_reward))+ 1))
            #self.current_node.imm_score = list(np.max(np.array(self.current_node.imm_score), np.divide(np.array(max_reward), 1+np.abs(np.array(max_reward)))))
            self.current_node = self.current_node.parent

    def search(self, n_step, epsilon=0.1, loop=10, gamma=0.90, rep_file=None, isLoadTree = False):
        if isLoadTree:
            self.root = ParseParetoSelectMCTS.load_tree(self = self, dataDir = hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"])
        self.set_repnode(rep_file=rep_file)
        #n_step = max(self.pareto.n_step, n_step)

        while self.step < n_step:
            self.step += 1
            print("--- step %d ---" % self.step)
            print("MAX_SCORE:", self.max_score)
            if self.n_valid+self.n_invalid == 0:
                valid_rate = 0
            else:
                valid_rate = self.n_valid/(self.n_valid+self.n_invalid)
            print("Validity rate:", valid_rate)
            self.select()
            self.expand(epsilon=epsilon, loop=loop, gamma=gamma)
            if len(self.next_token) != 0:
                self.simulate()
                self.backprop()
                #self.logging()
            self.save_tree(hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"])
            self.pareto.n_step = n_step
            self.pareto.save_pareto(hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"])

    def search_time(self, start_time, time_limit_sec=24*60*60, epsilon=0.1, loop=10, gamma=0.90, rep_file=None, isLoadTree = False):
        if isLoadTree:
            self.root = ParseParetoSelectMCTS.load_tree(self = self, dataDir = hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"])
        self.set_repnode(rep_file=rep_file)
        elapsed_time = 0
        one_epoch_time = 0
        while elapsed_time + one_epoch_time < time_limit_sec:
            time_begin = time.time()
            print(f"--- elapsed:{elapsed_time}\tremain:{time_limit_sec - elapsed_time} ---")
            print("MAX_SCORE:", self.max_score)
            if self.n_valid+self.n_invalid == 0:
                valid_rate = 0
            else:
                valid_rate = self.n_valid/(self.n_valid+self.n_invalid)
            print("Validity rate:", valid_rate)
            self.select()
            self.expand(epsilon=epsilon, loop=loop, gamma=gamma)
            if len(self.next_token) != 0:
                self.simulate()
                self.backprop()
            self.save_tree(hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"])
            elapsed_time = time.time() - start_time
            one_epoch_time = max(time.time() - time_begin, one_epoch_time)
        self.pareto.save_pareto(hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"])

    def set_repnode(self, rep_file=None):
        if len(rep_file) > 0:
            for smiles in read_smilesset(hydra.utils.get_original_cwd()+rep_file):
                n = ParentNode(smiles, cfg=self.Config)
                self.root.add_Node(n)
                c = NormalNode("&", cfg=self.Config)
                n.add_Node(c)
        else:
            for i in range(self.l_replace+1):
                for j in range(len(self.init_smiles)-i+1):
                    infix = self.init_smiles[j:j+i]
                    prefix = "".join(self.init_smiles[:j])
                    suffix = "".join(self.init_smiles[j + i:])

                    sc = prefix + "(*)" + suffix
                    mol_sc = Chem.MolFromSmiles(sc)
                    if mol_sc is not None:
                        n = ParentNode(prefix + "(*)" + suffix, cfg=self.Config)
                        self.root.add_Node(n)
                        c = NormalNode("&", cfg=self.Config)
                        n.add_Node(c)

    def save_tree(self, dir_path):
        """save tree to dir_path
        Arg: dirpath: path to dataDir
        
        save to dataDir/output/tree_save/tree{}.csv
        Return null
        """
        import shutil,os
        shutil.rmtree(dir_path+"output/tree_save")
        os.mkdir(dir_path+"output/tree_save")
        for i in range(len(self.root.children)):
            stack = []
            stack.extend(self.root.children[i].children)
            sc = self.root.children[i].original_smiles
            score = [self.root.children[i].cum_score]
            ids = [self.root.children[i].id]
            parent_id = [self.root.id]
            children_id = [str([c.id for c in self.root.children[i].children])]
            infix = [str(self.root.children[i].path)]
            rollout_smiles = [sc]
            rollout_score = [[-10000 for c in range(self.root.dimension)]]
            tokens = [self.root.children[i].token]
            c_s = [self.root.children[i].c]
            depths = [self.root.children[i].depth]
            visits = [self.root.children[i].visit]

            while len(stack) > 0:
                c = stack.pop(-1)
                for gc in c.children:
                    stack.append(gc)

                # save information
                score.append(c.cum_score)
                ids.append(c.id)
                parent_id.append(c.parent.id)
                ch_id = [gc.id for gc in c.children]
                children_id.append(str(ch_id))
                infix.append(str(c.path))
                rollout_smiles.append(c.rollout_result[0])
                rollout_score.append(c.rollout_result[1])
                tokens.append(c.token)
                c_s.append(c.c)
                depths.append(c.depth)
                visits.append(c.visit)

            df = pd.DataFrame(columns=["ID", "Score", "P_ID", "C_ID", "Infix", "Rollout_SMILES", "Rollout_Score", "Token", "C", "depth","visit"])
            df["ID"] = ids
            df["Score"] = score
            df["P_ID"] = parent_id
            df["C_ID"] = children_id
            df["Infix"] = infix
            df["Rollout_SMILES"] = rollout_smiles
            df["Rollout_Score"] = rollout_score
            df["Token"] = tokens
            df["c"] = c_s
            df["depth"] = depths
            df["visit"] = visits

            df.to_csv(dir_path+f"./output/tree_save/tree{i}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
        """ stack = []
        stack.extend(self.root.children)
        sc = ""
        score = [self.root.cum_score]
        ids = [-1]
        parent_id = [-1]
        children_id = [str([c.id for c in self.root.children])]
        infix = [sc]
        rollout_smiles = ["Scaffold"]
        rollout_score = [-10000]

        while len(stack) > 0:
            c = stack.pop(-1)
            for gc in c.children:
                stack.append(gc)

            # save information
            score.append(c.cum_score)
            ids.append(c.id)
            parent_id.append(c.parent.id)
            ch_id = [str(gc.id) for gc in c.children]
            children_id.append(",".join(ch_id))
            infix.append("".join(c.path))
            rollout_smiles.append(c.rollout_result[0])
            rollout_score.append(c.rollout_result[1])

        df = pd.DataFrame(columns=["ID", "Score", "P_ID", "C_ID", "Infix", "Rollout_SMILES", "Rollout_Score"])
        df["ID"] = ids
        df["Score"] = score
        df["P_ID"] = parent_id
        df["C_ID"] = children_id
        df["Infix"] = infix
        df["Rollout_SMILES"] = rollout_smiles
        df["Rollout_Score"] = rollout_score

        df.to_csv(dir_path+f"/tree{0}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC) """

    def logging(self, compound, scores):
        """logging output compounds 
        Input: 
            - compound: valid compound 
            - scores: reward scores correspond to the compound
        Return: 
            - Nothing        
        """
        dataDir = hydra.utils.get_original_cwd()+self.Config["mcts"]["data_dir"]
        with open(dataDir + "present/scores.txt", "a") as f:
            """ outStr = ""
            for score in scores:
                f.write(f"{score},")
            f.write("\n") """
            f.write(",".join(str(score) for score in scores)+"\n")
        with open(dataDir+"present/ligands.txt", "a") as f:
            f.write(compound+"\n")

        pass

    def load_tree(self, dataDir: str):
        """load tree from dir/output/tree_save/tree{}.csv
        Input: 
            - dataDir: path to dir

        Return
            - rootnode
        """
        self.root = RootNode(c=1/np.sqrt(2), cfg = self.Config)
        files = 0
        nodeDict = dict()
        parentDict = dict()
        childDict = dict()

        while True:
            if not os.path.exists(dataDir+f"./output/tree_save/tree{files}.csv"):
                break
            df = pd.read_csv(dataDir+f"./output/tree_save/tree{files}.csv")

            for i in range(len(df)):
                #df = pd.DataFrame(columns=["ID", "Score", "P_ID", "C_ID", "Infix", "Rollout_SMILES", "Rollout_Score", "Token","C"])
                scores = "".join(df["Score"][i])[1:-1].split(",")
                scores = list(map(lambda x:float(x), scores))

                childIds = "".join(df["C_ID"][i])[1:-1].split(",")
                childIds = list(map(lambda x: int(x), childIds)) if '' not in childIds else []
                
                if df["Token"][i] == "&&":
                    n = RootNode(c = df["C"][i], cfg=self.Config)
                elif df["Token"][i] == "SCFD":
                    n = ParentNode(scacffold=df["Rollout_SMILES"][i], c = df["C"][i], cfg=self.Config)
                else:
                    n = NormalNode(df["Token"][i], c=df["C"][i], cfg=self.Config)
                path = "".join(df["Infix"][i])[1:-1].split(",")
                path = list(map(lambda x:str(x).replace("\'","").replace(" ", ""), path))
                n.path = path
                n.id = int(df["ID"][i])
                global id
                id = max(id, n.id+1)
                self.total_nodes = id
                n.cum_score = scores
                n.depth = df["depth"][i]# (columns=["ID", "Score", "P_ID", "C_ID", "Infix", "Rollout_SMILES", "Rollout_Score", "Token", "C", "depth","visit"])
                n.visit = df["visit"][i]
                rollout_smiles = df["Rollout_SMILES"][i]
                rollout_scores = "".join(df["Rollout_Score"][i])[1:-1].split(",")
                rollout_scores = list(map(lambda x: float(x), rollout_scores))
                n.rollout_result = (rollout_smiles, rollout_scores)

                nodeDict[int(df["ID"][i])] = n
                parentDict[int(df["ID"][i])] = df["P_ID"][i]
                childDict[int(df["ID"][i])] = childIds
            files+=1
        if files>0:
            for i in nodeDict.keys():
                if nodeDict[i].id == -1:
                    self.root = nodeDict[i]
                elif parentDict[i] == -1 :
                    nodeDict[i].parent = self.root
                    self.root.children.append(nodeDict[i])
                else:
                    nodeDict[i].parent = nodeDict[parentDict[i]]
                for j in childDict[i]:
                    nodeDict[i].children.append(nodeDict[j])
            print(f"Loaded last {dataDir} with {files} csv file(s) ")
        else:
            print("NO FILES LOADED")
        return self.root


                

# add for PMOO
class Pareto():
    """ParetoFront Class
    """
    def __init__(self, front=[], compounds=[], cfg:DictConfig =None):
        """Constructor
        
            Input
                - front: reward score vectors in pareto front
                - compounds: compounds in pareto front
        """
        self.CONFIG = cfg
        self.front = front if front is not None else [[0 for i in range(len(self.CONFIG["reward"]["reward_list"]))]]
        self.compounds = compounds
        self.n_step = 0
    def initialization_front(self):
        if len(self.front) == 0:
            self.front = [[0 for i in range(len(self.CONFIG["reward"]["reward_list"]))]]
            self.compounds = ["cc"]

    def dominated(self, m):
        """Is point m dominated in reward space?
        
            Input
                - m: reward vector to decide dominated or not

            Return
                - True: Dominated
                - False: Non-dominated
        """
        if len(self.front) == 0:
            return True
        
        for p in self.front:
            if np.any(np.array(p) < np.array(m)):
                return True
            if np.all(np.array(p) >= np.array(m)):
                return False
        return False
    
    def update(self, scores, compound, dataDir):
        """Update Pareto front with new compound

            Input:
                - scores: reward score vector
                - compound: thinking compound
                - dataDir: save to outputDir

            Return:
                - None
        """
        del_list = []
        "--Is 'scores' better than current pareto front?---"
        for k in range(len(self.front)):
            """ for i in range(len(self.front[k])):
                if(self.front[k][i] >= scores[i]):
                    flag = False """
            if np.all(np.less(np.array(self.front[k]), np.array(scores))):
                del_list.append(k-len(del_list))
        for i in range(len(del_list)):
            del self.front[del_list[i]]
            del self.compounds[del_list[i]]
        self.front.append(scores)
        self.compounds.append(compound)
        #dataDir = hydra.utils.get_original_cwd()+OmegaConf.structured(Config)["mcts"]["data_dir"]
        with open(dataDir+"present/output.txt", 'a') as f:
            f.write(f"pareto size:{len(self.front)}\n")
            f.write(f"Updated pareto front\n{self.front}\n")
            f.write(f"Pareto Ligands\n{self.compounds}\n")
            f.write(f"Time:{time.asctime(time.localtime(time.time()))}\n")
        print(f"pareto size:{len(self.front)}")
    
    @staticmethod
    def from_dict(_filename, cfg: DictConfig):
        """Pareto front backup from files
            WARNING: you should check _filename file exists
            Input:
                - _filename: filepath of pareto.json

            Return:
                - Initialized Pareto front
        """
        with open(_filename, 'r') as f:
            _set_json = json.load(f)
            new_pareto = Pareto(front= _set_json['front'], compounds= _set_json['compounds'], cfg =cfg)
        print("Loaded Pareto Fronts")
        new_pareto.initialization_front()
        return new_pareto

    def wcal(self, ucb, reward, cnf):
        """Calculate HyperVolume
            Input: 
                - ucb: Upper Confidence Bound
                - reward: Thinking Reward Vector

            Return:
                - HyperVolume

        """
        
        hv = self._hvcal(ucb, cnf)
        if self.dominated(reward):
            return hv - self.distance(ucb)
        else:
            return hv
    
    def _getAverage(self):
        return list(np.array(self.front).mean(axis=0))

    def distance(self, ucb):
        """ Get sqrt distance
            Input:
                - ucb: Upper Confidence Bound
            
            Return:
                - sqrt distance toward average point of Pareto Area.
        """
        avg = self._getAverage()
        distance = 0
        for i in range(len(avg)):
            distance += pow(avg[i] - ucb[i],2)
        return np.sqrt(distance)


    def _hvcal(self, ucb, cnf):
        """Calculate Hypervolume Indicator
            Input:
                - ucb: Upper Confidence Bound

            Return:
                - HyperVolume Indicator
        """
        if len(self.front) == 0:
            return 0
        _pareto_temp = copy.deepcopy(self.front)
        _pareto_temp.append(ucb)
        #print(_pareto_temp,ucb, "line 739")
        for i in range(len(_pareto_temp)):
            for j in range(len(_pareto_temp[0])):
                if(_pareto_temp[i][j]>0):
                    _pareto_temp[i][j] *= -1
                else:
                    _pareto_temp[i][j] = -0.00000000000000001
        hv = hypervolume(_pareto_temp)
        ref_point = list(np.zeros_like(_pareto_temp[0]))
        try:
            hvnum = hv.compute(ref_point)
        except:
            with open(hydra.utils.get_original_cwd()+cnf["mcts"]["data_dir"]+"./present/hverror_output.txt", 'a') as f:
                print(time.asctime( time.localtime(time.time()) ),file=f)
                print(self.front,file=f)
            return 0
        return hvnum
    
    def _hv_prepare(self, _pareto_temp):
        for i in range(len(_pareto_temp)):
            for j in range(len(_pareto_temp[0])):
                if(_pareto_temp[i][j]>0):
                    _pareto_temp[i][j] *= -1
                else:
                    _pareto_temp[i][j] = -0.00000000000000001
        return _pareto_temp
    
    def save_pareto(self, dataDir):
        with open(dataDir+'present/pareto.json','w') as f:
            json.dump(self.__dict__, f, indent=4, separators=(',', ': '), default = self.__def_serialize)

    def __def_serialize(self, obj):
        if isinstance(obj, object):
            return None
        raise TypeError("Unsupport")


    def greatest_contributor(self)->int:
        _pareto_temp = copy.deepcopy(self.front)
        _pareto_temp = self._hv_prepare(_pareto_temp)
        hv = hypervolume(_pareto_temp)
        ref_point = list(np.zeros_like(_pareto_temp[0]))
        ind = hv.greatest_contributor(ref_point)
        return ind
        






@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    """--- constant ---"""
    vocab = VOCABULARY

    """--- input smiles ---"""
    start_smiles_list = read_smilesset(hydra.utils.get_original_cwd() + cfg["mcts"]["in_smiles_file"])

    for n, start_smiles in enumerate(start_smiles_list):
        n_valid = 0
        n_invalid = 0
        gen = {}
        mcts = None

        """--- MCTS ---"""
        model = RolloutNetwork(len(vocab))
        model_ver = cfg["mcts"]["model_ver"]
        model.load_state_dict(torch.load(hydra.utils.get_original_cwd() + cfg["mcts"]["model_dir"]
                                         + f"model-ep{model_ver}.pth",  map_location=torch.device('cpu')))

        reward = getReward(name=cfg["mcts"]["reward_name"],cfg=instantiate(cfg))#, init_smiles=start_smiles)
        rewards = getRewards(nameList=cfg["reward"]["reward_list"],cfg=instantiate(cfg))
        for r in rewards:
            if isinstance(r, DockingReward):
                r.dataDir = hydra.utils.get_original_cwd()+cfg["mcts"]["data_dir"]
        pareto = Pareto.from_dict(hydra.utils.get_original_cwd()+cfg["mcts"]["data_dir"]+"/present/pareto.json", cfg)
        input_smiles = start_smiles
        """ start = time.time()
        for i in range(cfg["mcts"]["n_iter"]):
            #mcts = ParseSelectMCTS(input_smiles, model=model, vocab=vocab, Reward=reward,
            #                       max_seq=cfg["mcts"]["seq_len"], step=cfg["mcts"]["n_step"] * i,
            #                       n_valid=n_valid, n_invalid=n_invalid, c=cfg["mcts"]["ucb_c"], max_r=reward.max_r)
            mcts = ParseParetoSelectMCTS(input_smiles, model=model, vocab=vocab, Reward=rewards,
                                   max_seq=cfg["mcts"]["seq_len"], step=cfg["mcts"]["n_step"] * i,
                                   n_valid=n_valid, n_invalid=n_invalid, c=cfg["mcts"]["ucb_c"], max_r=reward.max_r, Pareto = pareto, Config = cfg)
            mcts.search(n_step=cfg["mcts"]["n_step"] * (i + 1), epsilon=0, loop=10, rep_file=cfg["mcts"]["rep_file"], isLoadTree=cfg["mcts"]["isLoadTree"])
            reward.max_r = mcts.max_score
            n_valid += mcts.n_valid
            n_invalid += mcts.n_invalid
            gen = sorted(mcts.valid_smiles.items(), key=lambda x: x[1], reverse=True)
            input_smiles = gen[0][0] if len(gen)>0 else start_smiles
        end = time.time()
        print("Elapsed Time: %f" % (end-start)) """
        for i in range(cfg["mcts"]["n_iter"]):
            mcts = ParseParetoSelectMCTS(
                input_smiles,
                model=model,
                vocab=vocab,
                Reward=rewards,
                max_seq=cfg["mcts"]["seq_len"], 
                n_valid=n_valid,
                n_invalid=n_invalid,
                c=cfg["mcts"]["ucb_c"],
                max_r=reward.max_r,
                Pareto = pareto,
                Config = cfg)
            mcts.search_time(
                start_time = time.time(),
                time_limit_sec=cfg["mcts"]["time_limit_sec"]//cfg["mcts"]["n_iter"],
                epsilon=0,
                loop=10,
                rep_file=cfg["mcts"]["rep_file"],
                isLoadTree=cfg["mcts"]["isLoadTree"])
            ind = pareto.greatest_contributor()
            __scr = copy.deepcopy(pareto.front[ind])
            __smi = copy.deepcopy(pareto.compounds[ind])
            pareto.initialization_front()
            pareto.front.append(__scr)
            pareto.compounds.append(__smi)
            input_smiles = __smi
            print(f"Finished step {i}, best {__smi}, score{__scr}")

        generated_smiles = pd.DataFrame(columns=["SMILES", "Rewards", "Imp", "MW"])#, "step"])
        start_reward = []
        for reward in rewards:
            start_reward.append(reward.reward(start_smiles))
        for kv in mcts.valid_smiles.items():
            #step, smi = kv[0].split(":")
            #step = int(step)
            smi = kv[0]
            try:
                w = Descriptors.MolWt(Chem.MolFromSmiles(smi))
            except:
                w = 0

            generated_smiles.at[smi.rstrip('\n'), "SMILES"] = smi
            generated_smiles.at[smi.rstrip('\n'), "Rewards"] = kv[1]
            generated_smiles.at[smi.rstrip('\n'), "Imp"] = np.array(kv[1]) - np.array(start_reward)
            generated_smiles.at[smi.rstrip('\n'), "MW"] = w
            #generated_smiles.at[smi.rstrip('\n'), "step"] = step

        generated_smiles = generated_smiles.sort_values("Rewards", ascending=False)
        generated_smiles.to_csv(hydra.utils.get_original_cwd() +
                                cfg["mcts"]["out_dir"] + "No-{:04d}-{}.csv".format(n, start_smiles), index=False)
        with open(hydra.utils.get_original_cwd()+cfg["mcts"]["data_dir"]+"/input/next.smi","w") as f:
            f.write(input_smiles)


if __name__ == "__main__":
    main()


