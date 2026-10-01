import pytest
from feedback_loop import DistanceCache, JointFeedbackLoop

def test_distance_cache():
    def mock_astar(start, goal):
        if start == (0,0,0) and goal == (10,10,10):
            return None, 0.0 # Unreachable
        return [(start, goal)], 5.0

    cache = DistanceCache(A_star_func=mock_astar, max_cost=1000.0)
    
    dist = cache.get_distance((0,0,0), (1,1,1))
    assert dist == 5.0
    
    cache.update_mappo_distance((0,0,0), (1,1,1), 4.2)
    dist2 = cache.get_distance((0,0,0), (1,1,1))
    assert dist2 == 4.2
    
    dist3 = cache.get_distance((0,0,0), (10,10,10))
    assert dist3 == 1000.0
    
def test_schedule_parameterized():
    loop = JointFeedbackLoop(distance_cache=None, mappo_retrain_freq=500)
    assert loop.should_retrain_mappo() == False
    loop.ppo_episodes_since_retrain = 500
    assert loop.should_retrain_mappo() == True
