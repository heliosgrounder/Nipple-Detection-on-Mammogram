import itertools

class Utils:
    def __init__(self):
        pass

    def calculate_intersection(self, squares):
        x_min = max(min(square[0][0], square[1][0]) for square in squares)
        y_min = max(min(square[0][1], square[1][1]) for square in squares)
        x_max = min(max(square[0][0], square[1][0]) for square in squares)
        y_max = min(max(square[0][1], square[1][1]) for square in squares)
        
        if x_min < x_max and y_min < y_max:
            return ((x_min, y_min), (x_max, y_max))
        return None
    
    def calculate_center(self, squares):
        x_sum = sum(square[0][0] + square[1][0] for square in squares)
        y_sum = sum(square[0][1] + square[1][1] for square in squares)
        count = len(squares) * 2
        return (x_sum / count, y_sum / count)
    
    def distance_to_center(self, point, center):
        return ((point[0] - center[0])**2 + (point[1] - center[1])**2)**0.5
    
    def calculate_score(self, intersection, center, num_squares, total_squares):
        if not intersection:
            return float('-inf')
        
        intersection_center = (
            (intersection[0][0] + intersection[1][0]) / 2,
            (intersection[0][1] + intersection[1][1]) / 2
        )
        
        distance = self.distance_to_center(intersection_center, center)
        area = (intersection[1][0] - intersection[0][0]) * (intersection[1][1] - intersection[0][1])
        
        max_distance = self.distance_to_center((0, 0), center)
        normalized_distance = 1 - (distance / max_distance)
        max_area = (center[0] * 2) * (center[1] * 2)
        normalized_area = area / max_area
        
        square_weight = num_squares / total_squares
        
        score = (normalized_distance * 0.4 + normalized_area * 0.3 + square_weight * 0.3)
        
        return score
    
    def find_optimal_intersection(self, squares):
        center = self.calculate_center(squares)
        total_squares = len(squares)
        best_intersection = None
        best_score = float('-inf')
        
        for i in range(2, total_squares + 1):
            for combination in itertools.combinations(squares, i):
                intersection = self.calculate_intersection(combination)
                score = self.calculate_score(intersection, center, i, total_squares)
                
                if score > best_score:
                    best_score = score
                    best_intersection = intersection
        
        return best_intersection, best_score