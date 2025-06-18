import numpy as np
import itertools
import math

# 周波数
f = 2.4e9
# 光速
c = 2.99792458e8
# 波長
λ = c / f
# 波数
k = (2 * np.pi) / λ

#Triangleクラス
class Triangle():
    def __init__(self, verts:np.ndarray, name:int, envi_type:int):
        """面の情報

        Args:
            verts (np.ndarray): 頂点のリスト
            name (int): 面の名前
            envi_type (int): 環境の番号
        """
        #面の番号
        self.name = name

        #面の頂点
        self.a = verts[0]
        self.b = verts[1]
        self.c = verts[2]
        
        #ベクトル変換
        self.ab_vec = self.b - self.a
        self.ac_vec = self.c - self.a

        #法線ベクトル(規格化)
        n_vec = np.cross(self.ab_vec, self.ac_vec)
        self.n_vec = n_vec / np.linalg.norm(n_vec)

        #法線ベクトルの向きの調整
        if (envi_type == 0 or envi_type == 2) and self.name in [1, 2, 4, 7, 8, 11]:
            self.n_vec = -1 * self.n_vec
        elif envi_type == 1 and self.name in [22, 24, 26, 8, 7, 21, 13, 10, 0, 2, 4, 15, 17, 19]:
            self.n_vec = -1 * self.n_vec

    def calc_mir_p(self, p:np.ndarray) -> np.ndarray:
        """鏡像点算出関数
        
        鏡像点を計算する関数

        Args:
            tri_verts   (list)      :三角形の頂点を格納している配列
            p           (np.ndarray)  :鏡像点を求めたい点の位置ベクトル
        
        Return:
            mir_p       (np.ndarray)  :鏡像点の位置ベクトル
        """

        #符号付き距離
        h = np.dot(self.n_vec, self.a)

        #鏡像点
        return (p - 2 * (np.dot(self.n_vec, p) - h) * self.n_vec)
    
    #交差判定
    def judge_intersection(self, start_p:np.ndarray, end_p:np.ndarray):
        """交差判定関数
        
        レイと面が交差するかを判定する関数
        
        Args:
            start_p (np.array)  :始点の位置ベクトル
            end_p   (np.array)  :終点の位置ベクトル
            
        Return:
            if 交差する
            p       (np.array)  :反射点
            n_vec   (np.array)  :面の法線ベクトル
            
            else
            -1      (int)       :交差しないという意味の数
        """
        #方向ベクトル
        v_vec = end_p - start_p

        #方向パラメータ
        if np.dot(self.n_vec, v_vec) == 0:
            return -1 #交差しない

        t = np.dot(self.n_vec, (self.a - start_p)) / np.dot(self.n_vec, v_vec)

        #ABCを含む無限平面との交点の位置ベクトル
        inf_plane_inter_p = start_p + t * v_vec

        #g = p - a
        g_vec = inf_plane_inter_p - self.a

        #ABCの外部かどうか
        denominator = (
            np.dot(self.ab_vec, self.ac_vec)**2 - 
            np.dot(self.ac_vec, self.ac_vec) *
            np.dot(self.ab_vec, self.ab_vec)
        )
        if denominator== 0:
            return -1
    
        factor_v = (np.dot(g_vec, self.ab_vec) * np.dot(self.ab_vec, self.ac_vec) - np.dot(g_vec, self.ac_vec) * np.dot(self.ab_vec, self.ab_vec)) / denominator
        if not (0 <= factor_v <= 1):
            return -1

        factor_u = (np.dot(g_vec, self.ac_vec) * np.dot(self.ab_vec, self.ac_vec) - np.dot(g_vec, self.ab_vec) * np.dot(self.ac_vec, self.ac_vec)) / denominator
        if not (factor_u >= 0 and (factor_v + factor_u) <= 1):
            return -1
        
        # 現時点でpはstart_pとend_pを通る直線と三角形の交点を指す．
        # なので，【start_p->end_p->p】の場合も通している
        p = self.a + factor_u * self.ab_vec + factor_v * self.ac_vec
        
        vec1 = p - start_p
        vec2 = p - end_p
        if np.dot(vec1, vec2) < 0:
            return [p, self.n_vec, np.linalg.norm(vec1)] # (p:反射点、n_vec：面の法線ベクトル、norm(vec1)：レイの長さ)
        else:
            return -1

#Surfaceクラス
class Surface():
    def __init__(self, envi_type:int):
        """環境の情報

        Args:
            envi_type (int): 環境の番号
        """
        #環境の番号
        self.envi_type = envi_type
        # 環境データ
        self.points, self.surf_list, self.surf_list_idxs = self.get_envi_data(envi_type)
        #三角形のリスト
        self.tri_surf = self.make_trianle(self.surf_list, envi_type)
        # 頂点の最小値，最大値
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = self.get_min_max(self.points)

    def get_envi_data(self, envi_type):
        #環境番号:0 -> 直方体(rev2)
        if envi_type == 0:
            #頂点データ
            points = np.array([
                [-487.5,  -290,    0],    # 0: 下面：左下
                [-487.5,   290,    0],    # 1: 下面：左上
                [ 487.5,   290,    0],    # 2: 下面：右上
                [ 487.5,  -290,    0],    # 3: 下面：右下
                [-487.5,  -290,  262.5],  # 4: 上面：左下
                [-487.5,   290,  262.5],  # 5: 上面：左上
                [ 487.5,   290,  262.5],  # 6: 上面：右上
                [ 487.5,  -290,  262.5]   # 7: 上面：右下
            ]) 

            # 表面を作る頂点の組み合わせ（内側に向いている法線ベクトルの逆から見て時計回り）
            surf_list = [
                [points[1], points[0], points[3], points[2]],   # (z=0)
                [points[4], points[5], points[6], points[7]],   # (z=262.5)
                [points[1], points[5], points[4], points[0]],   # (x=-487.5)
                [points[3], points[7], points[6], points[2]],   # (x=487.5)
                [points[2], points[6], points[5], points[1]],   # (y=290)
                [points[0], points[4], points[7], points[3]]    # (y=-290)
            ]
            
        #環境番号:1 -> 実験室データ
        elif envi_type == 1:
            #頂点データ
            points = np.array([[-487.25,  -290,      0],
                                [-487.25,   290,      0],
                                [ 182.25,   290,      0],
                                [ 182.25,   194.5,    0],
                                [ 263.25,   194.5,    0],
                                [ 263.25,   290,      0],
                                [ 487.25,   290,      0],
                                [ 487.25,  -290,      0],
                                [ 182.25,  -290,      0],
                                [ 263.25,  -290,      0],
                                [-487.25,  -290,    262.5],
                                [-487.25,   290,    262.5],
                                [ 182.25,   290,    262.5],
                                [ 182.25,   194.5,  262.5],
                                [ 263.25,   194.5,  262.5],
                                [ 263.25,   290,    262.5],
                                [ 487.25,   290,    262.5],
                                [ 487.25,  -290,    262.5],
                                [ 182.25,  -290,    262.5],
                                [ 263.25,  -290,    262.5]])
            
            #面の設定
            surf_list = [[points[0],points[1],points[2],points[8]],
                            [points[8],points[3],points[4],points[9]],
                            [points[9],points[5],points[6],points[7]],
                            [points[0],points[1],points[11],points[10]],
                            [points[0],points[7],points[17],points[10]],
                            [points[7],points[6],points[16],points[17]],
                            [points[6],points[16],points[15],points[5]],
                            [points[5],points[15],points[14],points[4]],
                            [points[4],points[14],points[13],points[3]],
                            [points[3],points[13],points[12],points[2]],
                            [points[2],points[12],points[11],points[1]],
                            [points[10],points[18],points[12],points[11]],
                            [points[18],points[19],points[14],points[13]],
                            [points[19],points[17],points[16],points[15]]]
            
        #環境番号:2 -> 立方体
        elif envi_type == 2:
            #頂点データ
            points = np.array([[-400.0,  -400.0,    0],
                                [-400.0,  -400.0,  400.0],
                                [ 400.0,  -400.0,    0],
                                [ 400.0,  -400.0,  400.0],
                                [-400.0,   400.0,    0],
                                [-400.0,   400.0,  400.0],
                                [ 400.0,   400.0,    0],
                                [ 400.0,   400.0,  400.0]])

            surf_list = [[points[0],points[1],points[3],points[2]],
                        [points[0],points[1],points[5],points[4]],
                        [points[4],points[5],points[7],points[6]],
                        [points[2],points[3],points[7],points[6]],
                        [points[1],points[3],points[7],points[5]],
                        [points[0],points[2],points[6],points[4]]]

        #環境番号:3 -> 飯塚 (複雑な直方体)
        elif envi_type == 3:
            #頂点データ
            points = np.array([
                [-10, -8, -5],
                [-10, -8,  5],
                [ 10, -8, -5],
                [ 10, -8,  5],
                [-10,  8, -5],
                [-10,  8,  5],
                [ 10,  8, -5],
                [ 10,  8,  5],
                [ -7, -1, -5],
                [ -7, -1,  3],
                [  7, -1, -5],
                [  7, -1,  3],
                [ -7,  1, -5],
                [ -7,  1,  3],
                [  7,  1, -5],
                [  7,  1,  3]
                ])

            surf_list = [
                [points[0], points[1], points[3], points[2]],
                [points[0], points[1], points[5], points[4]],  
                [points[2], points[3], points[7], points[6]],  
                [points[1], points[3], points[7], points[5]],  
                [points[4], points[5], points[7], points[6]],  
                [points[0], points[8], points[10], points[2]],  
                [points[2], points[10], points[14], points[6]],  
                [points[6], points[14], points[12], points[4]],  
                [points[0], points[12], points[8], points[4]],  
                [points[8], points[10], points[11], points[9]],
                [points[10], points[14], points[15], points[11]],
                [points[12], points[14], points[15], points[13]],
                [points[9], points[12], points[8], points[13]],
                [points[11], points[15], points[13], points[9]]
                ]

            surf_list_idxs = [
                [0, 1, 3, 2],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [1, 3, 7, 5],
                [4, 5, 7, 6],
                [0, 8, 10, 2],
                [2, 10, 14, 6],
                [6, 14, 12, 4],
                [0, 12, 8, 4],
                [8, 10, 11, 9],
                [10, 14, 15, 11],
                [12, 14, 15, 13],
                [9, 12, 8, 13],
                [11, 15, 13, 9]
                ]
            
        #環境番号:4 -> 直方体 (非対称)
        elif envi_type == 4:
            #頂点データ
            points = np.array([[-487.5,  -290,    0],
                                [-487.5,  -290,  262.5],
                                [ 487.5,  -290,    0],
                                [ 487.5,  -290,  262.5],
                                [-487.5,   290,    0],
                                [-487.5,   290,  262.5],
                                [ 487.5,   290,    0],
                                [ 487.5,   290,  262.5]])

            #表面を作る頂点の組み合わせ(順番大切)
            surf_list = [[points[0],points[1],points[3],points[2]],
                            [points[0],points[1],points[5],points[4]],
                            [points[4],points[5],points[7],points[6]],
                            [points[2],points[3],points[7],points[6]],
                            [points[1],points[3],points[7],points[5]],
                            [points[0],points[2],points[6],points[4]]]

            surf_list_idxs = [
                [0, 1, 3, 2],
                [0, 1, 5, 4],
                [4, 5, 7, 6],
                [2, 3, 7, 6],
                [1, 3, 7, 5],
                [0, 2, 6, 4]
            ]

        #環境番号:5 -> 飯塚 (簡単な直方体)
        elif envi_type == 5:
            #頂点データ
            points = np.array([[-10, -8, -5],
                            [-10, -8,  5],
                            [ 10, -8, -5],
                            [ 10, -8,  5],
                            [-10,  8, -5],
                            [-10,  8,  5],
                            [ 10,  8, -5],
                            [ 10,  8,  5],
                            ])

            #表面を作る頂点の組み合わせ(順番大切)
            surf_list = [[points[0],points[1],points[3],points[2]],
                            [points[0],points[1],points[5],points[4]],
                            [points[4],points[5],points[7],points[6]],
                            [points[2],points[3],points[7],points[6]],
                            [points[1],points[3],points[7],points[5]],
                            [points[0],points[2],points[6],points[4]]]

            surf_list_idxs = [
                [0, 1, 3, 2],
                [0, 1, 5, 4],
                [4, 5, 7, 6],
                [2, 3, 7, 6],
                [1, 3, 7, 5],
                [0, 2, 6, 4]
            ]

        return [points, surf_list, surf_list_idxs]
            
    def make_trianle(self, surf_list, envi_type):
        """三角形を作成する関数
        
        四角形を三角形に分割する。
        法線ベクトルの逆から見て、左下から右上への対角線で分割する。
        
        Args:
            surf_list: 四角形の頂点リスト
            envi_type: 環境の種類
        
        Returns:
            tri_surf: 三角形オブジェクトのリスト
        """
        tri_surf = []
        name = -1
        
        for surf_verts in surf_list:
            # 四角形の頂点を反時計回りに並べ替える
            # surf_verts[0]: 左下
            # surf_verts[1]: 左上
            # surf_verts[2]: 右上
            # surf_verts[3]: 右下
            
            name += 1
            # 三角形1: 左下-左上-右上
            tri1 = Triangle(np.array([surf_verts[0], surf_verts[1], surf_verts[2]]), name, envi_type)
            
            name += 1
            # 三角形2: 左下-右上-右下
            tri2 = Triangle(np.array([surf_verts[0], surf_verts[2], surf_verts[3]]), name, envi_type)
            
            tri_surf.append(tri1)
            tri_surf.append(tri2)
        
        return tri_surf

    def get_min_max(self, points):
        return [min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1]), min(points[:, 2]), max(points[:, 2])]

#Rayクラス
class Ray():
    def __init__(self, t_p:np.ndarray, r_p:np.ndarray, ref_p_list:list, mir_p_list:list, tra_p_list:list, ray_interaction_flag_list:list, refsurf_n_list:list, trasurf_n_vec_list:list, ante_polar:str, ref_surfs:list):
        """インスタンス

        Args:
            t_p (np.ndarray): 送信点
            r_p (np.ndarray): 受信点
            ref_p_list (list): 反射点のリスト
            mir_p_list (list): 鏡像点のリスト
            tra_p_list (list): 透過点のリスト
            ray_interaction_flag_list (list): 相互作用リスト
            refsurf_n_list (list): 反射面の法線ベクトルのリスト
            trasurf_n_vec_list (list): 透過面の法線ベクトルのリスト
            ante_polar (str): アンテナの偏波（'TE':垂直，'TM':水平）

        """
        #反射点のリスト
        self.ref_p_list = ref_p_list
        #鏡像点のリスト
        self.mir_p_list = mir_p_list
        #透過点のリスト
        self.tra_p_list = tra_p_list
        #レイの相互作用フラグ配列（０：反射、１：透過）
        self.ray_interaction_flag_list = ray_interaction_flag_list
        #反射面の法線ベクトルのリスト
        self.refsurf_n_list = refsurf_n_list
        #透過面の法線ベクトルのリスト
        self.trasurf_n_list = trasurf_n_vec_list
        #ルート
        self.route = self.calc_route(t_p, r_p, ref_p_list, tra_p_list, ray_interaction_flag_list)
        #送信電力 [W] ※silicons labsが最大10mW(BLEの最大出力電力は100 mW)
        self.Pt = 10e-3
        #送信アンテナのゲイン [倍] ※今回は 1 倍で固定
        self.Gt = 1
        #受信アンテナのゲイン [倍] ※今回は 1 倍で固定
        self.Gr = 1
        #距離 [m]
        self.ray_distance = 0
        #距離減衰
        self.distance_damping = 0
        #反射減衰
        self.reflection_damping = 0
        # 複素振幅
        self.comp_amp = 0
        #減衰項
        self.damping_term = 0
        # アンテナの偏波
        self.ante_polar = ante_polar
        # 経由した三角形オブジェクト
        self.ref_surfs = ref_surfs

    
    def calc_route(self, tp, rp, ref_p_list, tra_p_list, ray_interaction_flag_list):
        """経路の計算
        """
        route = [tp, rp]
        ref_index = 0
        tra_index = 0

        #反射点を一つずつ格納する
        for flag in ray_interaction_flag_list:
            if flag == 0:
                route.insert(-1, ref_p_list[ref_index])
                ref_index += 1
            else:
                route.insert(-1, tra_p_list[tra_index])
                tra_index += 1
        
        return route

    #入射角を計算関数
    def calc_angle_of_in(self, op:np.ndarray, refp:np.ndarray, refsurf_n_vec:np.ndarray):
        """入射角を求める関数

        Args:
            op (np.ndarray): 発信点
            refp (np.ndarray): 反射点
            ref_surf_n_vec (np.ndarray): 反射点の法線ベクトル」
        """
        #レイの方向ベクトル
        ray_v_vec = refp - op

        #レイの方向ベクトルのノルム
        ray_v_vec_norm = np.linalg.norm(ray_v_vec)
        #反射面の法線ベクトルのノルム
        ref_surf_n_vec_norm = np.linalg.norm(refsurf_n_vec)

        #レイの単位方向ベクトル
        ray_unit_v = ray_v_vec / ray_v_vec_norm
        #反射面の法線単位ベクトル
        refsurf_unit_n = refsurf_n_vec / ref_surf_n_vec_norm

        #内積
        dot = np.dot(ray_unit_v, refsurf_unit_n)

        #入射角[rad]
        cos = -1 * dot

        return cos
    
    #2層フレネル反射係数
    def calc_fresnel_refcoef(self, op, refp, refsurf_n_vec):
        #真空中の誘電率 [F/m]
        ε_vac = 8.854e-12
        # 搬送波周波数 [GHz]
        f = 2

        # 媒質0：空気
        a0 = 1
        b0 = 0
        c0 = 0
        d0 = 0
        ε0_real = a0 * (f**b0)
        ε0_imag = (c0 * (f**d0))/(2 * np.pi * f * 1e9 * ε_vac)
        ε0 = ε0_real - 1j * ε0_imag

        # 媒質1 : コンクリート
        a1 = 5.24
        b1 = 0
        c1 = 0.0462
        d1 = 0.7822
        ε1_real = a1 * (f**b1)
        ε1_imag = (c1 * (f**d1))/(2 * np.pi * f * 1e9 * ε_vac)
        ε1 = ε1_real - 1j * ε1_imag

        #入射角の余弦
        θ = self.calc_angle_of_in(op, refp, refsurf_n_vec)

        if self.ante_polar == 'TE':
            fresnel_vertical_R = (np.sqrt(ε0) * math.cos(θ) - np.sqrt(ε1 - ε0 * (math.sin(θ))**2)) / (np.sqrt(ε0) * math.cos(θ) + np.sqrt(ε1 - ε0 * (math.sin(θ))**2))
            return fresnel_vertical_R
        elif self.ante_polar == 'TM':
            fresnel_parallel_R = (ε1 * math.cos(θ) - np.sqrt(ε0 * (ε1 - ε0 * (math.sin(θ))**2))) / (ε1 * math.cos(θ) + np.sqrt(ε0 * (ε1 - ε0 * (math.sin(θ))**2)))
            return fresnel_parallel_R
        
    #レイの距離を算出
    def calc_ray_distance(self):
        #レイの距離計算
        for i in range(0, len(self.route)-1):
            self.ray_distance += np.linalg.norm(self.route[i+1] - self.route[i])
        #[cm]→[m]に変換
        self.ray_distance = self.ray_distance / 100.0

    # 複素振幅の計算
    def calc_comp_amp(self):
        # 反射係数の配列
        deccoef_list = np.array([])
        for i in range(len(self.refsurf_n_list)):
            # 反射係数
            R = self.calc_fresnel_refcoef(self.route[i], self.ref_p_list[i], self.refsurf_n_list[i])
            # 格納
            deccoef_list = np.append(deccoef_list, R)
        # 距離減衰
        self.distance_damping = 1 / self.ray_distance
        # 反射減衰
        self.reflection_damping = np.prod(deccoef_list)
        # 複素振幅
        # self.comp_amp = np.sqrt(self.Pt) * np.sqrt(self.Gt) * np.sqrt(self.Gr) * (λ / (4 * np.pi * self.ray_distance)) * self.reflection_damping
        self.comp_amp = np.sqrt(self.Pt) * np.sqrt(self.Gt) * np.sqrt(self.Gr) * (λ / (4 * np.pi * self.ray_distance))
        # 減衰項
        self.damping_term = (np.exp(0 - 1j * k * self.ray_distance) / self.ray_distance) * self.reflection_damping
    

    #最後の相互作用点取得
    def get_last_p(self):
        return self.route[-2]
    
#RayTracingクラス
class RayTracing():
    def __init__(self, N, surfs, t_p, r_p, ante_polor):
        self.N = N
        self.surfs = surfs
        self.t_p = t_p
        self.r_p = r_p
        self.ante_polor = ante_polor
        #結果を格納する配列
        self.rays_list = []
        self.rays_last_p_list = []
        self.rays_distance_list = []
        self.rays_comp_amp_list = []

    #処理を行うか判定
    def judge_do_or_pass(self, list:list) -> int:
        """処理を行うか判定

        Args:
            list (list): クラス変数

        Returns:
            int: 0:実行、1:スルー
        """
        if len(list) == 0:
            return 1
        else:
            return 0
        
    def ray_search(self):
        """レイの探索関数
        """
        if self.judge_do_or_pass(self.rays_list):
            for k in range(1, self.N+1):                
                #反射面の順列
                surf_perm = list(itertools.permutations(self.surfs, k))

                for search_surf in surf_perm:
                    #それぞれの経路の鏡像点の配列
                    mir_p_list_tmp = []
                    #それぞれの経路の反射点の配列
                    ref_p_list_tmp = []
                    #レイの相互作用フラグ配列（０：反射、１：透過）
                    ray_interaction_flag_list_tmp = []
                    #反射面の法線ベクトルの配列
                    refsurf_n_vec_list_tmp = []

                    #鏡像点の計算
                    for i in range(0, k):
                        #最初のみ送信点の鏡像点
                        if i == 0:
                            mir_p = search_surf[i].calc_mir_p(self.t_p)
                            mir_p_list_tmp.append(mir_p)
                        #鏡像点の鏡像点
                        else:
                            mir_p = search_surf[i].calc_mir_p(mir_p_list_tmp[i-1]) 
                            mir_p_list_tmp.append(mir_p)

                    #レイの存在の有無
                    for j in reversed(range(0, k)):
                        #最初は受信点と最後の鏡像点
                        if j == k-1:
                            result_exists_ref_p = search_surf[j].judge_intersection(mir_p_list_tmp[j], self.r_p)
                        #反射点と鏡像点
                        else:
                            result_exists_ref_p = search_surf[j].judge_intersection(mir_p_list_tmp[j], ref_p_list_tmp[0])

                        #反射点なしなら
                        if type(result_exists_ref_p) == int:
                            break
                        else:
                            #反射点あれば、反射点、反射面の法線ベクトル、相互作用リスト
                            ref_p_list_tmp.insert(0, result_exists_ref_p[0])
                            refsurf_n_vec_list_tmp.insert(0, result_exists_ref_p[1])
                            ray_interaction_flag_list_tmp.insert(0, 0)

                    #レイがあるなら
                    if type(result_exists_ref_p) != int:
                        #格納
                        self.rays_list.append(Ray(self.t_p, self.r_p, ref_p_list_tmp, mir_p_list_tmp, [], ray_interaction_flag_list_tmp, refsurf_n_vec_list_tmp, [], self.ante_polor, search_surf))
            else: # 直接波
                self.rays_list.insert(0, Ray(self.t_p, self.r_p, [], [], [], [], [], [], self.ante_polor, []))

    def get_route_list(self, rays_list:list)->list:
        """全レイの経路を取得する

        Args:
            rays_list (list): レイのリスト

        Returns:
            list: 前レイの経路リスト
        """
        route_list = []
        for i in range(len(rays_list)):
            route_list.append(rays_list[i].route)
        return route_list


    def calc_rays_distance(self):
        """各レイの距離を計算
        """
        if self.judge_do_or_pass(self.rays_distance_list):
            for ray in self.rays_list:
                ray.calc_ray_distance()
                self.rays_distance_list.append(ray.ray_distance)

    def calc_rays_last_p_list(self):
        """各レイの最後の点を取得
        """
        if self.judge_do_or_pass(self.rays_last_p_list):
            for ray in self.rays_list:
                self.rays_last_p_list.append(ray.get_last_p())

    def calc_rays_comp_amp(self):
        """減衰項の計算, 複素振幅の計算
        """
        if self.judge_do_or_pass(self.rays_comp_amp_list):
            for ray in self.rays_list:
                ray.calc_comp_amp()
                self.rays_comp_amp_list.append(ray.comp_amp)

    def remove_duplicate_rays(self, rays_list: list) -> list:
        """同じ経路を持つレイを削除する

        Args:
            rays_list (list): レイのリスト

        Returns:
            list: 重複を除去したレイのリスト
        """
        # 各レイのrouteを小数点以下5位まで丸めてタプルに変換（比較のため）
        route_tuples = [tuple(map(tuple, np.round(ray.route, decimals=5))) for ray in rays_list]
        # route_tuples = [tuple(map(tuple, ray.route)) for ray in rays_list]

        
        # 重複を除去するためのインデックスを保持
        unique_indices = []
        seen_routes = set()
        
        for i, route in enumerate(route_tuples):
            if route not in seen_routes:
                unique_indices.append(i)
                seen_routes.add(route)
        
        # 重複のないレイのリストを作成
        unique_rays = [rays_list[i] for i in unique_indices]

        return unique_rays

    def do_ray_tracing(self):
        """レイトレーシングまとめ
        """
        self.ray_search()
        self.rays_list = self.remove_duplicate_rays(self.rays_list)
        self.calc_rays_distance()
        self.calc_rays_last_p_list()
        self.calc_rays_comp_amp()
