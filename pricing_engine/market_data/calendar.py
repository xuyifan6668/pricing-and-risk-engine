"""Calendar and date utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta


class Calendar(ABC):
    """Business day calendar."""

    @abstractmethod
    def is_business_day(self, dt: date) -> bool:
        raise NotImplementedError

    def adjust(self, dt: date) -> date:
        while not self.is_business_day(dt):
            dt += timedelta(days=1)
        return dt


class WeekendCalendar(Calendar):
    def is_business_day(self, dt: date) -> bool:
        return dt.weekday() < 5
