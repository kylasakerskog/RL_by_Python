from enum import Enum
import numpy as np

# セルの位置を示すクラス
class State():
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

# 上下左右の行動を制限
class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2

# 環境の定義
# grid を受け取り，Statesに保存
class Environment():
    def __init__(self, grid, move_prob=0.8):
        """
        grid 変数の要素
          0  : 通常
          -1 : ダメージ (ゲーム終了)
          1  : 報酬 (ゲーム終了)
          9  : 壁
        """
        self.grid = grid
        self.agent_state = State()

        # デフォルトの報酬を負にすることで学習を早める
        self.default_reward = -0.04

        # 要求と違う位置に動く確率 (1 - move_prob)
        self.move_prob = move_prob
        self.reset

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])
    
    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    
    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # 壁セルは state に含まれない
                if self.grid[row][column] != 9:
                    states.append(State(row,column))
        return states

    # 遷移関数
    def transit_func(self, state, action):
        transition_probs = {}
        
        if not self.can_action_at(state):
            return transition_probs

        opposite_direction = Action(action.value * -1)

        # 選択した行動にはself.move_prob
        # 逆方向以外の行動には残りの確率を等分した確率が割り当てられる
        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    # 移動を担うメソッド．
    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()
        
        # 動く方向の定義
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # マスを超えた移動や壁への移動の禁止
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state
        
    def reward_func(self, state):
        reward = self.default_reward
        done = False

        # 報酬値と終了の是非を返す
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True

        return reward, done

    # 環境を外部から扱うための関数
    # 移動したエージェントの位置を初期化する関数
    def reset(self):
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    # エージェントから行動を受け取って，遷移関数・報酬関数を用いて次の遷移先と即時報酬を計算
    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done
    
    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        # 遷移関数の出力した確率値に沿って遷移先を選択
        print(next_states)
        print(probs)
        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
