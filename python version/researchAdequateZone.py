
import numpy as np
import statistics
import copy
import math
import multiprocessing


def research_adequate_zone_from_script(mkpts_0, mkpts_1, group_index=-1):
    category1, category2, category3, category4, category5, category6, category7, category8, category9, category10, category11, category12, category13, category14, category15, category16, category17, category18, category19, category20, summoy, nbre = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    yaxis_class = [[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700],[700,800],[800,900],[900,1000],[1000,1080]]
    xaxis_class = [[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700],[700,800],[800,900],[900,950],[950,1000],[1000,1100],[1100,1200],[1200,1300],[1300,1400],[1400,1500],[1500,1600],[1600,1700],[1700,1800],[1800,1900],[1900,2000]]
    sum_result = None
    class_avgs = []
    fixed_point = []
    average_xaxis = []
    average_yaxis = []
    filtered_setleft, filtered_setright = [], []
    k = 0
    i, avg_abs, avg_ord1, avg_ord2, ydiff, bad_frame = 0,0,0,0,0,0
    area, ydiff_left = [], []
    concatpoint, mean_ratio = None, None
    set_points = {}
    transformed = []
    points_class, intensities_class = [], []
    yleft_right, yright_left = 0, 0
    for y in range(len(mkpts_0)):
        if mkpts_0[y][0] > mkpts_1[y][0]:
            if mkpts_0[y][1] > mkpts_1[y][1]:
                filtered_setleft.append([mkpts_0[y], mkpts_1[y]])
                ydiff_left.append(mkpts_0[y][1] - mkpts_1[y][1])
                yleft_right += 1
            elif mkpts_0[y][1] < mkpts_1[y][1]:
                filtered_setright.append([mkpts_0[y], mkpts_1[y]])
                ydiff += mkpts_1[y][1] - mkpts_0[y][1]
                yright_left += 1
    ydiff = ydiff / yright_left if yright_left > 0 else 0
    ydiff_left = statistics.mean(ydiff_left) if ydiff_left else 0
    filtered_setleft = [m for m in filtered_setleft if m[0][1] - m[1][1] < ydiff_left]
    filtered_setright = [m for m in filtered_setright if m[1][1] - m[0][1] < ydiff]
    if yleft_right > yright_left:
        liste1 = [[pair[0][0], pair[1][0]] for pair in filtered_setleft]
        liste2 = [[pair[0][1], pair[1][1]] for pair in filtered_setleft]
        for elt in xaxis_class:
            xdiff = []
            class_points = []
            summoy1, summoy, nbre = 0, 0, 0
            for m, n in filtered_setleft:
                if m[0] >= elt[0] and m[0] < elt[1]:
                    xdiff.append(m[0] - n[0])
            try:
                avgxdiff = statistics.mean(xdiff)
            except statistics.StatisticsError:
                avgxdiff = 0
            average_xaxis.append([avgxdiff, len(xdiff)])
        data = [sublist[0] for sublist in average_xaxis if sublist[-1] > 5]
        gen_avg = statistics.mean(data) if data else 0
        diff_avg = []
        for j in range(len(average_xaxis) - 1):
            if average_xaxis[j][1] != 0 and average_xaxis[j+1][1] != 0:
                diff_avg.append([abs(average_xaxis[j][0] - average_xaxis[j+1][0]), j])
        n = len(data)
        var = sum((x - gen_avg) ** 2 for x in data) / n if n > 0 else 0
        ecart = math.sqrt(var)
        small_diff, succ = [], ecart / 2
        while len(small_diff) == 0:
            small_diff = [num for num in diff_avg if num[0] <= succ and average_xaxis[num[1]][1] > 5]
            succ += 5
        result = []
        current_group = []
        for element in small_diff:
            if not current_group or abs(element[1] - current_group[-1][1]) <= 1:
                current_group.append(element)
            else:
                result.append(current_group)
                current_group = [element]
        if current_group:
            result.append(current_group)
        diff_genavg = []
        area = [[] for _ in range(len(result))]
        for e in range(len(result)):
            area[e].append(xaxis_class[result[e][0][1]][0])
            area[e].append(xaxis_class[result[e][-1][1]+1][1])
        diff_genavg = []
        if len(area) > 1:
            filtered_set = [[] for _ in range(len(result))]
            for i in range(len(area)):
                data = [m[0] - n[0] for m, n in filtered_setleft if area[i][0] <= m[0] <= area[i][1]]
                n = len(data)
                class_avg = statistics.mean(data) if n > 0 else 0
                var = sum((x - class_avg) ** 2 for x in data) / n if n > 0 else 0
                ecart = math.sqrt(var)
                diff_genavg.append([area[i], abs(class_avg - gen_avg) / n if n > 0 else 0, [class_avg, ecart]])
                diff_genavg = sorted(diff_genavg, key=lambda x: x[1])
            area = [m[0] for m in diff_genavg]
            filtered_setcopy = []
            for i in range(len(area)):
                error = diff_genavg[i][-1][1] / 2
                class_avg = diff_genavg[i][-1][0]
                while len(filtered_set[i]) == 0 and error < diff_genavg[i][-1][1] / 2 + 10:
                    filtered_set[i] = [[m, n, m[0] - n[0] - class_avg] for m, n in filtered_setleft if area[i][0] <= m[0] <= area[i][1] and class_avg - error < m[0] - n[0] < class_avg + error]
                    error += 5
                if len(filtered_set[i]) > 0:
                    filtered_set[i] = sorted(filtered_set[i], key=lambda x: x[-1])
                    filtered_set[i] = [elt[:2] for elt in filtered_set[i]]
                    filtered_setcopy.append(filtered_set[i])
                    class_avgs.append(class_avg)
                    set_points[str(area[i])] = filtered_set[i]
            try:
                mean_ratio = statistics.mean([class_avgs[0] - m for m in class_avgs[1:] if class_avgs[0] - m > 0 and class_avgs[0] - m <= 100])
            except statistics.StatisticsError:
                mean_ratio = None
            try:
                concatpoint = filtered_setcopy[0][int(len(filtered_setcopy) / 2)][:2]
                filtered_setcopy_for_detection = [item for sublist in filtered_setcopy for item in sublist]
                from warpImages import safe_extract_transformed_points
                transformed = safe_extract_transformed_points(filtered_setcopy, group_index)
            except IndexError:
                print("pas de points")
        else:
            filtered_set = []
            data = [m[0] - n[0] for m, n in filtered_setleft if area[0][0] <= m[0] <= area[0][1]]
            n = len(data)
            class_avg = statistics.mean(data) if n > 0 else 0
            var = sum((x - class_avg) ** 2 for x in data) / n if n > 0 else 0
            ecart = math.sqrt(var)
            error = ecart / 2
            while error < ecart / 2 + 10:
                filtered_set = [[m, n, m[0] - n[0] - class_avg] for m, n in filtered_setleft if area[0][0] <= m[0] <= area[0][1] and class_avg - error < m[0] - n[0] < class_avg + error]
                error += 5
            class_avgs.append(class_avg)
            set_points[str(area[0])] = filtered_set
            try:
                filtered_set = sorted(filtered_set, key=lambda x: x[-1])
                concatpoint = filtered_set[int(len(filtered_set) / 2)][:2]
                transformed = [[(float(m[0]), float(m[1])), (float(n[0]), float(n[1]))] for (m, n, _) in filtered_set]
            except IndexError:
                print("pas de points")
    else:
        for elt in xaxis_class:
            xdiff = []
            summoy1, summoy, nbre = 0, 0, 0
            for m, n in filtered_setright:
                if elt[0] <= m[0] < elt[1]:
                    xdiff.append(m[0] - n[0])
            try:
                avgxdiff = statistics.mean(xdiff)
                average_xaxis.append([avgxdiff, len(xdiff)])
            except statistics.StatisticsError:
                avgxdiff = 0
                average_xaxis.append([avgxdiff, len(xdiff)])
        data = [sublist[0] for sublist in average_xaxis if sublist[-1] > 5]
        gen_avg = statistics.mean(data) if data else 0
        diff_avg = []
        for j in range(len(average_xaxis) - 1):
            if average_xaxis[j][1] != 0 and average_xaxis[j+1][1] != 0:
                diff_avg.append([abs(average_xaxis[j][0] - average_xaxis[j+1][0]), j])
        small_diff, succ = [], 10
        while len(small_diff) == 0:
            small_diff = [num for num in diff_avg if num[0] <= succ and average_xaxis[num[1]][1] > 5]
            succ += 5
        copysmall_diff = copy.deepcopy(small_diff)
        result = []
        current_group = []
        for element in small_diff:
            if not current_group or abs(element[1] - current_group[-1][1]) <= 1:
                current_group.append(element)
            else:
                result.append(current_group)
                current_group = [element]
        if current_group:
            result.append(current_group)
        area = [[] for _ in range(len(result))]
        diff_genavg = []
        for e in range(len(result)):
            area[e].append(xaxis_class[result[e][0][1]][0])
            area[e].append(xaxis_class[result[e][-1][1]+1][1])
        diff_genavg = []
        if len(area) > 1:
            filtered_set = [[] for _ in range(len(result))]
            for i in range(len(area)):
                data = [m[0] - n[0] for m, n in filtered_setright if area[i][0] <= m[0] <= area[i][1]]
                n = len(data)
                class_avg = statistics.mean(data) if n > 0 else 0
                var = sum((x - class_avg) ** 2 for x in data) / n if n > 0 else 0
                ecart = math.sqrt(var)
                diff_genavg.append([area[i], abs(class_avg - gen_avg) / n if n > 0 else 0, [class_avg, ecart]])
                diff_genavg = sorted(diff_genavg, key=lambda x: x[1])
            area = [m[0] for m in diff_genavg]
            class_avgs = []
            filtered_setcopy = []
            for i in range(len(area)):
                error = diff_genavg[i][-1][1] / 2
                class_avg = diff_genavg[i][-1][0]
                while len(filtered_set[i]) == 0 and error < diff_genavg[i][-1][1] / 2 + 10:
                    filtered_set[i] = [[m, n, m[0] - n[0] - class_avg] for m, n in filtered_setright if area[i][0] <= m[0] <= area[i][1] and class_avg - error < m[0] - n[0] < class_avg + error]
                    error += 5
                if len(filtered_set[i]) > 0:
                    filtered_set[i] = sorted(filtered_set[i], key=lambda x: x[-1])
                    filtered_set[i] = [elt[:2] for elt in filtered_set[i]]
                    filtered_setcopy.append(filtered_set[i])
                    class_avgs.append(class_avg)
                    set_points[str(area[i])] = filtered_set[i]
            try:
                mean_ratio = statistics.mean([class_avgs[0] - m for m in class_avgs[1:] if class_avgs[0] - m > 0 and class_avgs[0] - m <= 100])
            except statistics.StatisticsError:
                mean_ratio = None
            try:
                concatpoint = filtered_setcopy[0][int(len(filtered_setcopy) / 2)][:2]
                from warpImages import safe_extract_transformed_points
                transformed = safe_extract_transformed_points(filtered_setcopy, group_index)
            except IndexError:
                print("pas de points")
        else:
            filtered_set = []
            data = [m[0] - n[0] for m, n in filtered_setright if area[0][0] <= m[0] <= area[0][1]]
            n = len(data)
            class_avg = statistics.mean(data) if n > 0 else 0
            var = sum((x - class_avg) ** 2 for x in data) / n if n > 0 else 0
            ecart = math.sqrt(var)
            error = ecart / 2
            while len(filtered_set) == 0 and error < ecart / 2 + 10:
                filtered_set = [[m, n, m[0] - n[0] - class_avg] for m, n in filtered_setright if area[0][0] <= m[0] <= area[0][1] and class_avg - error < m[0] - n[0] < class_avg + error]
                error += 5
            class_avgs.append(class_avg)
            set_points[str(area[0])] = filtered_set
            try:
                filtered_set = sorted(filtered_set, key=lambda x: x[-1])
                concatpoint = filtered_set[int(len(filtered_set) / 2)][:2]
                transformed = [[(float(m[0]), float(m[1])), (float(n[0]), float(n[1]))] for (m, n, _) in filtered_set]
            except IndexError:
                print("pas de points")
    return transformed


def research_wrapper(mkpts_0, mkpts_1, result_queue):
    try:
        transformed = research_adequate_zone_from_script(mkpts_0, mkpts_1)
        result_queue.put(("SUCCESS", transformed))
    except Exception as e:
        result_queue.put(("ERROR", str(e)))


def research_with_real_timeout(mkpts_0, mkpts_1, timeout=5):
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=research_wrapper, args=(mkpts_0, mkpts_1, result_queue))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    if not result_queue.empty():
        status, result = result_queue.get()
        if status == "SUCCESS":
            return result
        else:
            return None
    else:
        return None