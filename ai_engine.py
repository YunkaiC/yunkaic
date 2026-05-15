"""
争上游卡牌游戏 - AI出牌决策模块

AI策略拆分为独立函数，便于后续优化：
1. 主动出牌策略 (ai_play_free)
2. 被动跟牌策略 (ai_play_follow)
3. 各种辅助决策函数

设计原则：
- 被动跟牌：优先选最小能压住的牌，尽量不拆炸弹
- 主动出牌：优先出组合牌减少手牌，避免无意义拆散大组合
- 手牌少时提高进攻性

难度级别：
- easy: 30%随机出牌，极少使用炸弹，无终局优化
- normal: 标准策略，合理使用炸弹
- hard: 更积极使用炸弹，终局优化更激进，但并非"永远最优"
  （受限于搜索深度和启发式规则，hard模式仍有改进空间）
"""
import random
from collections import Counter
from card import Card
from game_logic import (
    Pattern, HandType, recognize_pattern, compare_hands,
    find_all_valid_plays, _count_ranks, find_all_bombs,
    _get_bomb_ranks, BOMB_CONFIG
)
from constants import Rank


# 难度对应的进攻性系数
DIFFICULTY_AGGRESSIVENESS = {
    'easy': 0.3,
    'normal': 0.5,
    'hard': 0.8,
}


class AIEngine:
    """
    AI出牌决策引擎

    所有决策函数均为独立函数，便于单独测试和优化

    支持三种难度：
    - easy: 30%随机出牌，极少使用炸弹，无终局优化
    - normal: 标准策略，合理使用炸弹和终局优化
    - hard: 更积极使用炸弹和终局优化，进攻性更高
      （注意：hard模式并非"永远最优"，仍基于启发式规则，
      存在搜索深度和策略覆盖面的局限）
    """

    def __init__(self, aggressiveness: float = None, difficulty: str = 'normal'):
        """
        Args:
            aggressiveness: 进攻性系数 0.0-1.0，值越大越激进（已弃用，用difficulty代替）
            difficulty: 难度级别 'easy'/'normal'/'hard'
        """
        self.difficulty = difficulty
        if aggressiveness is not None:
            self.aggressiveness = aggressiveness
        else:
            self.aggressiveness = DIFFICULTY_AGGRESSIVENESS.get(difficulty, 0.5)

    def decide(self, cards: list[Card], last_play: Pattern | None,
               is_free_turn: bool, game_state: dict = None) -> tuple[list[Card], bool]:
        """
        AI决策入口

        Args:
            cards: 当前手牌
            last_play: 上一次出牌的牌型
            is_free_turn: 是否为自由出牌轮
            game_state: 游戏状态信息（用于更高级的决策）

        Returns:
            (出牌列表, 是否选择不要)
            如果选择不要，返回 ([], True)
        """
        if not cards:
            return [], not is_free_turn

        # Easy模式：30%概率随机出牌
        if self.difficulty == 'easy' and random.random() < 0.3:
            play = self._random_valid_play(cards, last_play, is_free_turn)
            if play is not None:
                return play, False
            # 随机出牌失败，回退到正常逻辑

        # 终局优化：如果手牌<=3张，检查是否能一次性出完
        # Easy模式跳过终局优化
        if self.difficulty != 'easy' and len(cards) <= 3:
            all_cards_play = self._can_play_all_remaining(cards, last_play, is_free_turn)
            if all_cards_play is not None:
                return all_cards_play, False

        if is_free_turn or last_play is None:
            play_cards = self.ai_play_free(cards, game_state)
            return play_cards, False
        else:
            return self.ai_play_follow(cards, last_play, game_state)

    # ==================== 主动出牌策略 ====================

    def ai_play_free(self, cards: list[Card], game_state: dict = None) -> list[Card]:
        """
        主动出牌策略（自由出牌）

        策略优先级：
        1. 如果手牌只剩1张或1对，直接出
        2. 终局检查：手牌<=6时检查是否能一次出完
        3. 优先出顺子、连对、飞机等组合牌（减少手牌数量最快）
        4. 出三带一/三带二（比纯三张更有效率）
        5. 出单张或对子（从小到大，避开炸弹点数）
        6. 保留炸弹作为后手

        Args:
            cards: 手牌
            game_state: 游戏状态

        Returns:
            要出的牌列表
        """
        if not cards:
            return []

        remaining = len(cards)

        # 如果只剩1张，直接出
        if remaining == 1:
            return cards[:]

        # 如果只剩2张且是合法牌型，直接出
        if remaining == 2:
            pattern = recognize_pattern(cards)
            if pattern.hand_type != HandType.INVALID:
                return cards[:]
            # 否则出最小的单张
            return [min(cards, key=lambda c: c.sort_key)]

        # 终局优化：检查是否能一次性出完
        if self.difficulty != 'easy' and remaining <= 6:
            all_play = self._can_play_all_remaining(cards, None, True)
            if all_play is not None:
                return all_play

        # 获取进攻性调整（手牌越少越激进）
        aggr = self._adjusted_aggressiveness(remaining)

        # 分析手牌结构
        rank_count = _count_ranks(cards)
        bomb_ranks = _get_bomb_ranks(rank_count)
        all_plays = find_all_valid_plays(cards, None)

        # 分类可用牌型
        combos = []      # 组合牌（顺子、连对、飞机等）
        triples = []     # 三张
        triple_ones = []  # 三带一
        triple_twos = []  # 三带二
        pairs = []       # 对子
        singles = []     # 单张
        bombs = []       # 炸弹

        for play in all_plays:
            if play.hand_type in (HandType.STRAIGHT, HandType.STRAIGHT_PAIR,
                                  HandType.TWO_STRAIGHT_PAIR,
                                  HandType.STRAIGHT_TRIPLE, HandType.PLANE_ONE,
                                  HandType.PLANE_TWO):
                combos.append(play)
            elif play.hand_type == HandType.TRIPLE:
                triples.append(play)
            elif play.hand_type == HandType.TRIPLE_ONE:
                triple_ones.append(play)
            elif play.hand_type == HandType.TRIPLE_TWO:
                triple_twos.append(play)
            elif play.hand_type == HandType.PAIR:
                pairs.append(play)
            elif play.hand_type == HandType.SINGLE:
                singles.append(play)
            elif play.hand_type in (HandType.BOMB, HandType.ROCKET):
                bombs.append(play)

        # Hard模式：更早使用炸弹
        bomb_threshold_free = 8 if self.difficulty == 'hard' else 5

        # 策略1：如果手牌较少且有炸弹，考虑使用炸弹快速清场
        if remaining <= bomb_threshold_free and bombs:
            non_bomb_cards = self._cards_without_bombs(cards, bombs)
            if not non_bomb_cards:
                # 只有炸弹了，出炸弹
                return self._select_smallest_bomb(bombs)
            # 如果炸弹出完后剩余牌能一次出完，先出炸弹
            bomb_cards = self._select_smallest_bomb(bombs)
            if bomb_cards:
                remaining_after_bomb = self._remove_cards(cards, bomb_cards)
                if not remaining_after_bomb or self._can_play_all_remaining(
                        remaining_after_bomb, None, True):
                    return bomb_cards

        # 策略1b：手牌较少且激进，考虑炸弹清场
        if remaining <= 8 and aggr >= 0.7 and bombs:
            non_bomb_cards = self._cards_without_bombs(cards, bombs)
            if not non_bomb_cards:
                return self._select_smallest_bomb(bombs)

        # 策略2：优先出组合牌（顺子、连对等）
        if combos:
            best_combo = self._select_best_combo(combos, aggr, rank_count)
            if best_combo:
                return best_combo.cards

        # 策略3：出三带一/三带二（比纯三张更高效）
        if triple_twos:
            triple_twos.sort(key=lambda p: p.main_rank)
            return triple_twos[0].cards

        if triple_ones:
            triple_ones.sort(key=lambda p: p.main_rank)
            return triple_ones[0].cards

        # 策略4：出三张（如果不需要带牌）
        if triples:
            triples.sort(key=lambda p: p.main_rank)
            return triples[0].cards

        # 策略5：出对子（从小到大，避开炸弹点数——已在find_all_valid_plays中过滤）
        if pairs:
            pairs.sort(key=lambda p: p.main_rank)
            return pairs[0].cards

        # 策略6：出单张（从小到大，避开炸弹点数——已在find_all_valid_plays中过滤）
        if singles:
            singles.sort(key=lambda p: p.main_rank)
            return singles[0].cards

        # 策略7：出炸弹
        if bombs:
            return self._select_smallest_bomb(bombs)

        # 兜底：出最小的牌
        cards_sorted = sorted(cards, key=lambda c: c.sort_key)
        return [cards_sorted[0]]

    # ==================== 被动跟牌策略 ====================

    def ai_play_follow(self, cards: list[Card], last_play: Pattern,
                       game_state: dict = None) -> tuple[list[Card], bool]:
        """
        被动跟牌策略

        策略：
        1. 优先选择能压住的最小合法牌
        2. 尽量不拆炸弹
        3. 手牌少时提高进攻性
        4. 没有合适牌时选择不要

        Args:
            cards: 手牌
            last_play: 上一次出牌的牌型
            game_state: 游戏状态

        Returns:
            (出牌列表, 是否选择不要)
        """
        remaining = len(cards)
        aggr = self._adjusted_aggressiveness(remaining)

        # 找到所有能压住上家的合法出牌
        valid_plays = find_all_valid_plays(cards, last_play)

        if not valid_plays:
            # 无法压住，选择不要
            return [], True

        # 分类：普通牌型 vs 炸弹
        normal_plays = [p for p in valid_plays
                        if p.hand_type not in (HandType.BOMB, HandType.ROCKET)]
        bomb_plays = [p for p in valid_plays
                      if p.hand_type in (HandType.BOMB, HandType.ROCKET)]

        # 如果有普通牌型可以压
        if normal_plays:
            # 选择最小的能压住的牌（按主牌点数排序，点数相同按消耗牌数多优先）
            normal_plays.sort(key=lambda p: (p.main_rank, -len(p.cards)))

            # 尽量不拆炸弹：优先选不涉及炸弹点数的牌
            # 注意：find_all_valid_plays 已经应用了炸弹不可拆分规则，
            # 所以 normal_plays 中的牌不应该包含炸弹点数的牌。
            # 但仍做一次安全检查，以防枚举不完整
            non_bomb_breaking = self._filter_non_bomb_breaking(normal_plays, cards)

            chosen_cards = None
            if non_bomb_breaking:
                chosen_cards = non_bomb_breaking[0].cards
            elif aggr > 0.6:
                # 如果手牌少且激进，允许出牌（正常出牌不应拆炸弹，但兜底）
                chosen_cards = normal_plays[0].cards
            elif game_state and self._should_use_bomb(game_state, remaining):
                # 如果对手手牌也很少，用炸弹阻止
                if bomb_plays:
                    chosen_cards = self._select_smallest_bomb(bomb_plays)
                else:
                    chosen_cards = normal_plays[0].cards
            else:
                # 普通跟牌
                chosen_cards = normal_plays[0].cards

            # 安全验证：确保出牌合法且能压过上家
            if chosen_cards:
                validated = self._validate_play(chosen_cards, last_play)
                if validated:
                    return chosen_cards, False
                # 验证失败，尝试其他选项
                for play in normal_plays:
                    if self._validate_play(play.cards, last_play):
                        return play.cards, False
                # 所有普通出牌都验证失败，尝试炸弹或不出
                if bomb_plays and self._should_use_bomb(game_state, remaining):
                    for bplay in bomb_plays:
                        if self._validate_play(bplay.cards, last_play):
                            return bplay.cards, False
                return [], True
            return [], True

        # 只有炸弹可以压
        if bomb_plays:
            use_bomb = False

            # Hard模式：更容易使用炸弹
            if self.difficulty == 'hard':
                use_bomb = True
            else:
                # 条件1：对手手牌<=2，用炸弹阻止对手获胜
                opponent_min = 999
                if game_state:
                    opponent_min = game_state.get('opponent_min_cards', 999)
                if opponent_min <= 2:
                    use_bomb = True

                # 条件2：自己手牌<=5且有炸弹，即使保守也考虑使用
                if remaining <= 5:
                    use_bomb = True

                # 条件3：激进模式或原有的炸弹策略
                if aggr > 0.5 or self._should_use_bomb(game_state, remaining):
                    use_bomb = True

                # Easy模式：极少使用炸弹（10%概率）
                if self.difficulty == 'easy' and use_bomb:
                    use_bomb = random.random() < 0.1

            if use_bomb:
                # 选择最小的能赢的炸弹
                chosen_cards = self._select_smallest_winning_bomb(bomb_plays, last_play)
                if chosen_cards:
                    validated = self._validate_play(chosen_cards, last_play)
                    if validated:
                        return chosen_cards, False
                # 炸弹验证失败，尝试其他炸弹
                for bplay in bomb_plays:
                    if self._validate_play(bplay.cards, last_play):
                        return bplay.cards, False
                return [], True

            # 保守策略，不浪费炸弹
            return [], True

        # 无牌可出
        return [], True

    # ==================== 辅助决策函数 ====================

    def _random_valid_play(self, cards: list[Card], last_play: Pattern | None,
                           is_free_turn: bool) -> list[Card] | None:
        """
        随机选择一个合法出牌（Easy模式用）

        Args:
            cards: 手牌
            last_play: 上一次出牌的牌型
            is_free_turn: 是否为自由出牌轮

        Returns:
            随机合法出牌列表，如果没有合法出牌返回None
        """
        target = last_play if not is_free_turn else None
        valid_plays = find_all_valid_plays(cards, target)
        if not valid_plays:
            return None
        # 过滤掉炸弹（easy模式极少用炸弹）
        non_bomb_plays = [p for p in valid_plays
                          if p.hand_type not in (HandType.BOMB, HandType.ROCKET)]
        if non_bomb_plays:
            return random.choice(non_bomb_plays).cards
        # 只剩炸弹时，5%概率使用
        if random.random() < 0.05:
            return random.choice(valid_plays).cards
        return None

    def _validate_play(self, cards: list[Card], last_play: Pattern) -> bool:
        """
        验证出牌是否合法且能压过上家

        这是安全检查，防止因枚举边界情况导致AI出非法牌。

        Args:
            cards: 要出的牌
            last_play: 上一次出牌

        Returns:
            True如果出牌合法
        """
        if not cards:
            return False

        pattern = recognize_pattern(cards)
        if pattern.hand_type == HandType.INVALID:
            return False

        if last_play is not None:
            result = compare_hands(pattern, last_play)
            if result <= 0:
                return False

        return True

    def _adjusted_aggressiveness(self, remaining_cards: int) -> float:
        """
        根据剩余手牌数量调整进攻性

        手牌越少，进攻性越高

        Args:
            remaining_cards: 剩余手牌数

        Returns:
            调整后的进攻性系数
        """
        if remaining_cards <= 3:
            return min(1.0, self.aggressiveness + 0.4)
        elif remaining_cards <= 6:
            return min(1.0, self.aggressiveness + 0.2)
        elif remaining_cards <= 10:
            return self.aggressiveness
        else:
            return max(0.0, self.aggressiveness - 0.1)

    def _select_best_combo(self, combos: list[Pattern],
                           aggr: float, rank_count: Counter = None) -> Pattern | None:
        """
        选择最佳组合牌出牌

        策略：
        - 激进时优先出长组合（快速减手牌）
        - 保守时优先出短组合（保留长组合的后手）
        - 避免出会拆散其他有效组合的牌

        Args:
            combos: 可用的组合牌列表
            aggr: 当前进攻性
            rank_count: 手牌点数统计

        Returns:
            最佳组合牌
        """
        if not combos:
            return None

        # 按消耗牌数排序（从多到少）
        combos.sort(key=lambda p: -len(p.cards))

        if aggr > 0.6:
            # 激进：出最长的组合
            return combos[0]
        else:
            # 保守：出最短的组合
            return combos[-1]

    def _select_smallest_bomb(self, bombs: list[Pattern]) -> list[Card]:
        """
        选择最小的炸弹

        优先选择张数最少且点数最小的炸弹，保留王炸作后手。

        Args:
            bombs: 可用炸弹列表

        Returns:
            炸弹牌列表
        """
        if not bombs:
            return []

        # 普通炸弹优先于王炸
        normal_bombs = [b for b in bombs if b.hand_type == HandType.BOMB]
        rockets = [b for b in bombs if b.hand_type == HandType.ROCKET]

        if normal_bombs:
            # 选张数最少、点数最小的
            normal_bombs.sort(key=lambda b: (b.bomb_size, b.main_rank))
            return normal_bombs[0].cards

        if rockets:
            # 普通王炸优先于双王炸
            single_rockets = [r for r in rockets if not r.is_double_rocket]
            double_rockets = [r for r in rockets if r.is_double_rocket]
            if single_rockets:
                return single_rockets[0].cards
            return double_rockets[0].cards

        return bombs[0].cards

    def _select_smallest_winning_bomb(self, bombs: list[Pattern],
                                       last_play: Pattern) -> list[Card]:
        """
        选择最小的能赢的炸弹

        在跟牌场景下，选择刚好能压过上家的最小炸弹，
        避免浪费更大的炸弹。

        Args:
            bombs: 可用炸弹列表
            last_play: 上一次出牌

        Returns:
            炸弹牌列表
        """
        if not bombs:
            return []

        # 过滤出能赢的炸弹
        winning_bombs = []
        for b in bombs:
            if compare_hands(b, last_play) > 0:
                winning_bombs.append(b)

        if not winning_bombs:
            return []

        # 普通炸弹优先于王炸（王炸留作后手）
        normal_winning = [b for b in winning_bombs if b.hand_type == HandType.BOMB]
        rocket_winning = [b for b in winning_bombs if b.hand_type == HandType.ROCKET]

        if normal_winning:
            # 选张数最少、点数最小的能赢炸弹
            normal_winning.sort(key=lambda b: (b.bomb_size, b.main_rank))
            return normal_winning[0].cards

        if rocket_winning:
            # 普通王炸优先于双王炸
            single_rockets = [r for r in rocket_winning if not r.is_double_rocket]
            double_rockets = [r for r in rocket_winning if r.is_double_rocket]
            if single_rockets:
                return single_rockets[0].cards
            return double_rockets[0].cards

        return []

    def _filter_non_bomb_breaking(self, plays: list[Pattern],
                                  cards: list[Card]) -> list[Pattern]:
        """
        过滤掉会拆炸弹的出牌

        虽然 find_all_valid_plays 已经应用了炸弹不可拆分规则，
        但此函数作为额外的安全检查层，确保万无一失。

        如果出牌使用了某炸弹点数的部分牌（非全部作为炸弹使用），
        则视为"拆炸弹"。

        Args:
            plays: 可用出牌列表
            cards: 手牌

        Returns:
            不拆炸弹的出牌列表
        """
        rank_count = _count_ranks(cards)
        bomb_ranks = _get_bomb_ranks(rank_count)

        if not bomb_ranks:
            return plays

        safe_plays = []
        for play in plays:
            # 炸弹/王炸本身不算拆炸弹
            if play.hand_type in (HandType.BOMB, HandType.ROCKET):
                safe_plays.append(play)
                continue

            # 检查该出牌是否使用了炸弹点数的牌
            play_rank_count = _count_ranks(play.cards)
            breaks_bomb = False
            for rank in play_rank_count:
                if rank in bomb_ranks:
                    breaks_bomb = True
                    break

            if not breaks_bomb:
                safe_plays.append(play)

        return safe_plays

    def _should_use_bomb(self, game_state: dict | None,
                         my_remaining: int) -> bool:
        """
        判断是否应该使用炸弹

        考虑因素：
        - 对手手牌数量（对手快出完时用炸弹阻止）
        - 自己手牌数量（自己快出完时用炸弹冲刺）

        Args:
            game_state: 游戏状态
            my_remaining: 自己剩余手牌数

        Returns:
            是否应该使用炸弹
        """
        if game_state is None:
            return my_remaining <= 5

        # 对手快出完时用炸弹（≤2张时更积极）
        opponent_cards = game_state.get('opponent_min_cards', 999)
        if opponent_cards <= 2:
            return True
        if opponent_cards <= 3:
            return True

        # 自己快出完时用炸弹冲刺
        if my_remaining <= 4:
            return True

        return False

    def _can_play_all_remaining(self, cards: list[Card],
                                 last_play: Pattern | None = None,
                                 is_free_turn: bool = True) -> list[Card] | None:
        """
        检查是否可以一次性出完所有手牌

        终局优化：当手牌较少时，检查是否所有剩余牌构成一个合法牌型。
        如果是，直接出完以获胜。跟牌时还需验证能否压过上家。

        Args:
            cards: 手牌列表
            last_play: 上一次出牌的牌型（跟牌时使用）
            is_free_turn: 是否为自由出牌

        Returns:
            如果可以一次出完，返回出牌列表；否则返回None
        """
        if not cards:
            return None

        pattern = recognize_pattern(cards)
        if pattern.hand_type == HandType.INVALID:
            return None

        # 跟牌时需验证能否压过上家
        if last_play is not None and not is_free_turn:
            if compare_hands(pattern, last_play) <= 0:
                return None

        return cards[:]

    def _cards_without_bombs(self, cards: list[Card],
                              bombs: list[Pattern]) -> list[Card]:
        """
        获取不包含炸弹牌的手牌

        使用 (suit, rank, deck_id) 元组作为键进行安全比较，
        避免直接用 Card 对象放入 set 可能出现的潜在问题。

        Args:
            cards: 手牌
            bombs: 炸弹列表

        Returns:
            不含炸弹牌的手牌
        """
        bomb_keys = set()
        for bomb in bombs:
            for c in bomb.cards:
                bomb_keys.add((c.suit, c.rank, c.deck_id))
        return [c for c in cards if (c.suit, c.rank, c.deck_id) not in bomb_keys]

    def _remove_cards(self, cards: list[Card], to_remove: list[Card]) -> list[Card]:
        """
        从手牌中移除指定牌

        使用 (suit, rank, deck_id) 元组作为键进行安全比较。

        Args:
            cards: 原手牌
            to_remove: 要移除的牌

        Returns:
            移除后的手牌
        """
        remove_keys = set((c.suit, c.rank, c.deck_id) for c in to_remove)
        return [c for c in cards if (c.suit, c.rank, c.deck_id) not in remove_keys]
