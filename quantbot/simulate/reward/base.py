from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from general.interfaces.reward_interface import RewardInterface


class Reward(RewardInterface):
    """默认奖励系统实现"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        persist_path = self.config.get('persist_path', 'artifacts/reward_logs.json')
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        # 单轮次内的决策轨迹；只在轮次结束时持久化一次
        self.trajectories: List[Dict[str, Any]] = []

        self.logger.info(f"奖励系统初始化完成，持久化路径: {self.persist_path}")

    def record_trajectory(
        self,
        timestamp: datetime,
        account,
        prompt: str,
        output: str,
    ) -> None:
        """记录单次Agent决策轨迹"""
        try:
            account_snapshot = None
            profit_loss_rate = None

            if hasattr(account, 'get_account_schema'):
                account_schema = account.get_account_schema()
                profit_loss_rate = account_schema.account_info.total_profit_loss_rate
                try:
                    account_snapshot = json.loads(account_schema.json())
                except Exception:
                    # 回退到dict + 默认序列化
                    account_snapshot = account_schema.dict()

            self.trajectories.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "prompt": prompt,
                    "output": output,
                    "account_snapshot": account_snapshot,
                    "profit_loss_rate": profit_loss_rate,
                    "score": None,
                }
            )

            self.logger.debug(
                "记录轨迹: time=%s, profit_loss_rate=%s",
                timestamp,
                profit_loss_rate,
            )
        except Exception as exc:
            self.logger.error(f"记录轨迹失败: {exc}")
            raise

    def calculate_score(self) -> None:
        """按照最后一次账户总收益率计算当轮得分"""
        if not self.trajectories:
            self.logger.warning("没有记录到任何轨迹，跳过得分计算")
            return

        # 以最后一条轨迹的收益率作为本轮得分
        last = self.trajectories[-1]
        score = last.get("profit_loss_rate", 0.0) or 0.0

        for trajectory in self.trajectories:
            trajectory["score"] = score

        self.logger.info(
            "本轮共 %d 条轨迹，收益率=%.4f，已更新得分",
            len(self.trajectories),
            score,
        )

    def persist_data(self) -> bool:
        """持久化当前轮的轨迹与得分"""
        try:
            record = {
                "trajectories": self.trajectories,
            }

            persisted_data: List[Any] = []
            if self.persist_path.exists():
                try:
                    with self.persist_path.open('r', encoding='utf-8') as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list):
                            persisted_data = loaded
                        elif loaded:
                            persisted_data = [loaded]
                except Exception as exc:
                    self.logger.warning(f"读取历史奖励数据失败，将覆盖文件: {exc}")

            persisted_data.append(record)

            with self.persist_path.open('w', encoding='utf-8') as f:
                json.dump(persisted_data, f, ensure_ascii=False, indent=2)

            self.logger.info("奖励数据已保存至 %s", self.persist_path)
            return True

        except Exception as exc:
            self.logger.error(f"奖励数据保存失败: {exc}")
            return False

