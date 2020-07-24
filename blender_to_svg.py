
bl_info = {
    "name" : "SVG Export",
    "author": "VUT",
    "version": (0, 1, 0),
    "blender": (2, 80, 0),
    "location": "File > Export",
    "description": "Export screen as SVG",
    "warning": "",
    "category": "Export"
}

import bpy
#from bpy_extras import view3d_utils
from mathutils import Vector
import time


class Vertex:
    
    def __init__( self, coords ):
        self.co = coords.copy()
        #self.world_co = coords_w.copy()
    
    #make object indexable
    def __getitem__( self, index ):
        return self.co[index]

    def __setitem__( self, index, value ):
        self.co[index] = value
        
    #math operations
    def __add__( self, n ):
        if isinstance( n, Vertex ):
            return Vector ( [ self.co[0] + n.co[0],
                              self.co[1] + n.co[1],
                              self.co[2] + n.co[2],
                              self.co[3] + n.co[3] ] )
                              
        elif isinstance( n, Vector ):
            return Vector ( [ self.co[0] + n[0],
                              self.co[1] + n[1],
                              self.co[2] + n[2],
                              self.co[3] + n[3] ] )


    def __sub__( self, n ):
        if isinstance( n, Vertex ):
            return Vector ( [ self.co[0] - n.co[0],
                              self.co[1] - n.co[1],
                              self.co[2] - n.co[2],
                              self.co[3] - n.co[3] ] )
                              
        elif isinstance( n, Vector ):
            return Vector ( [ self.co[0] - n[0],
                              self.co[1] - n[1],
                              self.co[2] - n[2],
                              self.co[3] - n[3] ] )
    
    
    def __mul__( self, n ):
        if isinstance( n, float ):
            return Vector( [ self.co[0] * n,
                             self.co[1] * n,
                             self.co[2] * n,
                             self.co[3] * n ] )


    def __truediv__ ( self, n ):
        if isinstance( n, float ):
            return Vector( [ self.co[0] / n,
                             self.co[1] / n,
                             self.co[2] / n,
                             self.co[3] / n ] )


    def __eq__ ( self, n ):
        if isinstance( n, Vector ):
            return abs( self.co[0] - n[0] ) <= 0.00005 and abs( self.co[1] - n[1] ) <= 0.00005 and abs( self.co[2] - n[2] ) <= 0.00005
        


class Intersection ( Vertex ):
    
    def __init__( self, coords, super_index, sub_index ):
        super().__init__( coords )
        self.super_index = super_index
        self.sub_index = sub_index


class Curve:
    
    def __init__ ( self, vertices, color ):
        self.vertices = vertices.copy() #indices to array of vertices
        self.color = color
        
    def draw_object( self, file, scene, res ):
        
        #file.write( f"""<!-- {self.id}--> \n""" )
        file.write( f"""<polyline points=\"""" )
        
        for vertex in self.vertices:
            v = scene.vertex_to_screen_coords( scene.vertices[vertex], res )
            file.write( f"""{round( v[0], 4 )},{round( v[1], 4 )} """ )
        
#        for v in self.vertices:
#            file.write( f"""{round( vertices[v][0], 4 )},{round( vertices[v][1], 4 )} """ )
        file.write( f"""\" fill="none" stroke="rgb({str(self.color[:3])[1:-1]})" stroke-width="2" stroke-linecap="round"/>\n""" )
        


class Triangle:
    
    def __init__( self, vertices, data, i, poly = None ):
        self.vertices = vertices.copy() #indices to array of vertices
        self.index = i      #index to array of triangles
        self.polygon = poly    #index to array of polygons
        self.normal = self.normal_vector_compute( data )
        self.d = self.d_param_compute( data )
        
    # n = ( B - A ) x ( C - A )
    def normal_vector_compute ( self, data ):
        BA = data[ self.vertices[1] ] - data[ self.vertices[0] ]
        CA = data[ self.vertices[2] ] - data[ self.vertices[0] ]

        return Vector( [BA[1]*CA[2] - BA[2]*CA[1],
                        BA[2]*CA[0] - BA[0]*CA[2],
                        BA[0]*CA[1] - BA[1]*CA[0]] )
                    
    # d = a*Ax + b*Ay + c*Az
    def d_param_compute ( self, data ):
        return (  self.normal[0] * ( data[ self.vertices[0] ][0] / data[ self.vertices[0] ][3] ) 
                + self.normal[1] * ( data[ self.vertices[0] ][1] / data[ self.vertices[0] ][3] )
                + self.normal[2] * ( data[ self.vertices[0] ][2] / data[ self.vertices[0] ][3] ) )

        

class Polygon:
    
    def __init__ ( self, vertices, poly_index, color, id ):
        self.vertices = vertices.copy() #indices to array of vertices
        self.tris = []  #indices to array of triangles
        self.poly_index = poly_index    #index to array of polygons
        self.color = color
        self.id = id
        self.elems_under = set()
        
        
    def draw_object( self, file, scene, res ):
        
        file.write( f"""<!-- {self.id}--> \n""" )
        file.write( f"""<polygon points=\"""" )
        
        for vertex in self.vertices:
            v = scene.vertex_to_screen_coords( scene.vertices[vertex], res )
            file.write( f"""{round( v[0], 4 )},{round( v[1], 4 )} """ )
            
        #float array to string
        poly_c = str( self.color[:3] )[1:-1]

        file.write( f"""\" fill="rgb({poly_c})" fill-opacity="{self.color[3]}" """ +
                    f"""stroke="rgb(0,0,0)" stroke-width="1" stroke-linejoin="round"/>\n""" )
                    
        
            
                    
        
class Scene:
    
    def __init__( self ):
        self.vertices = { (i + j if j == 1 else i + j + 3) : Vertex( [ 1.0*i,  1.0*j, 1.0, 1.0 ] ) for i in range(-1,2,2) for j in range(1,-2,-2) }
        self.vertices[2], self.vertices[3] = self.vertices[3], self.vertices[2]
        self.curves = list()
        self.triangles = list()
        self.polygons = list()
        self.sort_objects = list()
        self.camera = None
        
    
    
    
    def vertex_to_screen_coords ( self, vertex, res ):
                        
        v = vertex.co.copy()
        #to NDC space
        v = v / v[3]
        #to screen space
        v[0] = 0.5 * res[0] * ( 1.0 + v[0] )
        v[1] = 0.5 * res[1] * ( 1.0 + v[1] )
        
        return v



    def overlap ( self, f_poly, s_poly ):
        
        for axis in range(3):
            if ( all( self.vertices[vf][axis] - self.vertices[vs][axis] <= 0.0001 for vf in f_poly.vertices for vs in s_poly.vertices ) ):
                return True
                   
            if ( all( self.vertices[vf][axis] - self.vertices[vs][axis] >= -0.0001 for vf in f_poly.vertices for vs in s_poly.vertices ) ):
                return True
            
        return False                    
        
    
    
    
    
    def check_collision_clip_space ( self, elem_f, elem_s ):
        
        
        if isinstance( elem_f, Curve ) and isinstance( elem_s, Curve ):
            pass
        elif isinstance( elem_f, Curve ) and isinstance( elem_s, Polygon ):
            pass
        elif isinstance( elem_f, Polygon ) and isinstance( elem_s, Curve ):
            pass
        elif isinstance( elem_f, Polygon ) and isinstance( elem_s, Polygon ):
#            print('elem_f ', elem_f)
#            print('elem_s ', elem_s)
            for f_tri_i in elem_f.tris:
                f_tri = self.triangles[ f_tri_i ]
                
                for s_tri_i in elem_s.tris:
                    s_tri = self.triangles[ s_tri_i ]
#                    print('f_tri ', f_tri)
#                    print('s_tri ', s_tri)
                    res, in_tri, out_tri = self.triangles_overlap( f_tri, s_tri )
                    #if overlap in clip space
                    if res:
                        #cross edges
                        if not in_tri:
                            #find intersections
                            for inter_n in self.find_tri_tri_intersection_2d( f_tri, s_tri ):
                                inter_f, inter_s = self.get_intersections( inter_n, f_tri, s_tri )
                                if inter_f == inter_s:
                                    continue
                                elif inter_f[2] < inter_s[2]:
                                    return True
                                else:
                                    return False
                        #one tri in second
                        else:
                            temp_arr = in_tri.vertices.copy()
                            for i_vert in in_tri.vertices:
                                for o_vert in out_tri.vertices:
                                    i = self.vertices[i_vert]
                                    o = self.vertices[o_vert]
                                    if i.co == o.co:
                                        temp_arr.remove(i_vert)
                            try:
                                inter_hom = self.vertices[ temp_arr[0] ]
                            except IndexError:
                                inter_hom = self.vertices[ in_tri.vertices[0] ]
                            
                            inter = ( inter_hom / inter_hom[3] ).to_2d()
                            inter_f, inter_s = self.get_intersections( inter, f_tri, s_tri )
#                            print('inter_f ', inter_f)
#                            print('inter_s ', inter_s)
                            if inter_f == inter_s:
                                continue
                            elif inter_f[2] < inter_s[2]:
                                return True
                            else:
                                return False
                    
        return False


    
    def point_tri_overlap ( self, f_tri, s_tri ):
        
        #loop through points of second triangle
        for j in range(3):
            X_j = s_tri.vertices[j]
            X = self.vertices[X_j]
            X_n = ( X / X[3] ).to_2d()
            
            res = []
            #loop through edges of first triangle
            for i in range(3):
                C_i = f_tri.vertices[i - 2]
                A_i = f_tri.vertices[i - 1]
                B_i = f_tri.vertices[i]
                A = self.vertices[A_i]
                B = self.vertices[B_i]
                C = self.vertices[C_i]
                        
                A_n = ( A / A[3] ).to_2d()
                B_n = ( B / B[3] ).to_2d()
                C_n = ( C / C[3] ).to_2d()
                    
                # line equation
                # n.(X - A) = 0
                s = ( B_n - A_n )
                n = Vector( ( -s[1], s[0] ) ).normalized()
                
                opposite = n.dot( C_n - A_n )
                if opposite < 0.0:
                    n *= -1.0
                
                res.append( n.dot( X_n - A_n ) )
                
            if all( r > 0.0 for r in res ):
                yield X_n
                

                
        
    def tri_tri_overlap ( self, f_tri, s_tri ):
    
        sec_in_first = True
        #loop through edges of first triangle
        for i in range(3):
            C_i = f_tri.vertices[i - 2]
            A_i = f_tri.vertices[i - 1]
            B_i = f_tri.vertices[i]
            A = self.vertices[A_i]
            B = self.vertices[B_i]
            C = self.vertices[C_i]
                        
            A_n = ( A / A[3] ).to_2d()
            B_n = ( B / B[3] ).to_2d()
            C_n = ( C / C[3] ).to_2d()
                    
            # line equation
            # n.(X - A) = 0
            s = ( B_n - A_n )
            n = Vector( ( -s[1], s[0] ) ).normalized()
            
            opposite = n.dot( C_n - A_n )
            if opposite < 0.0:
                n *= -1.0
            
            #check all points of second triangle
            res = []
            for j in range(3):
                X_j = s_tri.vertices[j]
                X = self.vertices[X_j]
                X_n = ( X / X[3] ).to_2d()
                
                res.append( n.dot( X_n - A_n ) )
            
            if all( r <= 0.0 for r in res ):
                return False, False
            
            if all( r >= 0.0 for r in res ):
                sec_in_first = sec_in_first and True
            else:
                sec_in_first = sec_in_first and False
                
        return True, sec_in_first
    
    
    
    def triangles_overlap ( self, f_tri, s_tri ):
    
        res_1, inner_1 = self.tri_tri_overlap( f_tri, s_tri )
        res_2, inner_2 = self.tri_tri_overlap( s_tri, f_tri )
        
        in_tri = None
        out_tri = None
        if inner_1:
            in_tri = s_tri
            out_tri = f_tri
        elif inner_2:
            in_tri = f_tri
            out_tri = s_tri
    
        return res_1 and res_2, in_tri, out_tri


    
    def find_line_line_intersection_2d ( self, A_n, B_n, D_n, E_n ):
        
        #determinant
        D_00 = B_n[0] - A_n[0]
        D_01 = D_n[0] - E_n[0]
        D_10 = B_n[1] - A_n[1]
        D_11 = D_n[1] - E_n[1]
                
        #right side
        b_0 = D_n[0] - A_n[0]
        b_1 = D_n[1] - A_n[1]
                
        try:
            det = D_00 * D_11 - D_10 * D_01
            t =  ( b_0*D_11 - b_1*D_01 ) / det
            s =  ( D_00*b_1 - D_10*b_0 ) / det
        except ZeroDivisionError:
            return None
                
        if ( 0.0 < t < 1.0 and 0.0 < s < 1.0 ):
#            C_i = f_tri.vertices[i - 2]
#            C = self.vertices[C_i]
#            C_n = ( C / C[3] ).to_2d()
#                    print('f tri')
#                    print(A_n)
#                    print(B_n)
#                    print(C_n)
                    
#            F_i = s_tri.vertices[j - 2]
#            F = self.vertices[F_i]
#            F_n = ( F / F[3] ).to_2d()
#                    print('s tri')
#                    print(D_n)
#                    print(E_n)
#                    print(F_n)
                    
#                    print('t ', t)
#                    print('s ', s)
            return D_n + ( E_n - D_n ) * s
                
        return None
    
    
        
    def find_tri_tri_intersection_2d ( self, f_tri, s_tri ):
                
        for i in range(3):
            A_i = f_tri.vertices[i - 1]
            B_i = f_tri.vertices[i]
            A = self.vertices[A_i]
            B = self.vertices[B_i]
            
            A_n = ( A / A[3] ).to_2d()
            B_n = ( B / B[3] ).to_2d()
            
            for j in range(3):
                D_j = s_tri.vertices[j - 1]
                E_j = s_tri.vertices[j]
                D = self.vertices[D_j]
                E = self.vertices[E_j]
                
                D_n = ( D / D[3] ).to_2d()
                E_n = ( E / E[3] ).to_2d()
                
                inter = self.find_line_line_intersection_2d( A_n, B_n, D_n, E_n )
                if inter:
                    yield inter
                
        for vert in self.point_tri_overlap( f_tri, s_tri ):
            yield vert 
            
        for vert in self.point_tri_overlap( s_tri, f_tri ):
            yield vert
                
                    
            
    def find_cross_point_line_line ( self, A, B, C, D ):
        
        #determinant
        D_00 = B[0] - A[0]
        D_01 = D[0] - C[0]
        D_10 = B[1] - A[1]
        D_11 = D[1] - C[1]
        
        #right side
        b_0 = C[0] - A[0]
        b_1 = C[1] - A[1]
        
        try:
            det = D_00 * D_11 - D_10 * D_01
            t =  ( b_0*D_11 - b_1*D_01 ) / det
            s = -( D_00*b_1 - D_10*b_0 ) / det
        except ZeroDivisionError:
            return None
        
        if ( abs( t * ( B[2]-A[2] ) - s * ( D[2] - C[2] ) + ( A[2]-C[2] ) ) < 0.004 ):
            if( 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0 ):
                return C + ( D - C ) * s

        return None
    
    
    
    def get_intersections ( self, inter, f_tri, s_tri ):
                            
        near = self.camera.data.clip_start
        far  = self.camera.data.clip_end
    
        #line from near plane to far plane
        I_n = Vector( ( inter[0]*2*near, inter[1]*2*near, -2*near ) )
        I_f = Vector( ( inter[0]*far, inter[1]*far, far ) )
        R = I_f - I_n
    
        #find intersections with triangles and line
        n1 = f_tri.normal.normalized()
        d = n1.dot( self.vertices[ f_tri.vertices[0] ] )
        f = n1.dot( I_n )
        e = n1.dot( R )
        
        if ( e == 0.0 ):
            return None, None
        
        x = ( d - f ) / e
        if ( 0.0 < x and x < 1.0 ):
            #compute intersection
            inter_f = I_n + R * x
        
        
        n2 = s_tri.normal.normalized()
        d = n2.dot( self.vertices[ s_tri.vertices[0] ] )
        f = n2.dot( I_n )
        e = n2.dot( R )
        
        if ( e == 0.0 ):
            return None, None
        
        x = ( d - f ) / e
        if ( 0.0 < x and x < 1.0 ):
            #compute intersection
            inter_s = I_n + R * x
            
        return inter_f, inter_s
    
    
    
    def get_intersections_l ( self, inter, A, B, C, D ):
        
        near = self.camera.data.clip_start
        far  = self.camera.data.clip_end
        
        #line from near plane to far plane
        I_n = Vector( ( inter[0]*2*near, inter[1]*2*near, -2*near ) )
        I_f = Vector( ( inter[0]*far, inter[1]*far, far ) )
        
        #find intersections with line and line
        inter_f = self.find_cross_point_line_line( I_n, I_f, A, B )
        inter_s = self.find_cross_point_line_line( I_n, I_f, C, D )
        
        return inter_f, inter_s
       




# ------------------------------------------------------------------------------------------ #
#                                            MAIN                                            #
# ------------------------------------------------------------------------------------------ #
class SVG_export( bpy.types.Operator ):

    """Export to SVG Script"""
    bl_idname = "obj.to_svg"
    bl_label = "blender_to_svg"
    
    #def __init__(self):


    def execute(self, context):
        
        start_time = time.time()

        print('\n---------------------------------------------------------------------------')
        print('\t\t\tExport scene to SVG')
        print('---------------------------------------------------------------------------\n')
        
        b_scene = bpy.context.scene
        render = b_scene.render
        #get resolution from blender
        self.res = [ render.resolution_x, render.resolution_y ]
            
        #create empty scene
        scene = Scene()
            
        #get active camera
        scene.camera = b_scene.camera
        if not scene.camera:
            return { 'CANCELLED' }
        elif scene.camera.type != 'CAMERA':
            return { 'CANCELLED' }
        #print(camera.visible_get())

        if ( scene.camera.data.type == 'PERSP' ):
            vp_mat = self.compute_vp_matrix( scene.camera, render )
        else:
            return { 'CANCELLED' }
        

        #load object from scene
        #surfaces = [ o for o in b_scene.objects if o.type == 'SURFACE' ]
        #text = [ o for o in b_scene.objects if o.type == 'TEXT' ]
        
        
        
        
        #directional light L(x,y,z)
        scene.camera_loc = scene.camera.location
        up  = scene.camera.matrix_world.to_quaternion() @ Vector( (0.0, 1.0, 0.0) )
        dir = scene.camera.matrix_world.to_quaternion() @ Vector( (0.0, 0.0, -1.0) )
        scene.light_dir = ( ( dir.cross( up ) - dir ) + up ).normalized()
        
        print( 'Clipping elements' )
        #v - index to array of vertices
        #p - index to array of polygons
        obj_inds = { 'v': 4, 'p': 0 }
        
        #loop through visible objects in the scene
        for obj in ( obj for obj in b_scene.objects if obj.visible_get() ):
            if ( obj.type == 'CURVE' ):
                self.clip_curve( obj, scene, vp_mat, obj_inds )
            elif ( obj.type == 'MESH' ):
                self.clip_triangles( obj, scene, vp_mat, obj_inds )
        
        #clip polygons
        self.clip_polygons( scene )

        print("--- %s seconds ---" % (time.time() - start_time))                    



                
        
#        print ( '\n-----COLLISION TEST-----\n' )
        # CHECK COLLISIONS
        # POINT - LINE
#        for curve in reversed( curves ):
#            temp_verts = curve.vertices.copy()
#            end = len( curve.vertices )
#            for i in range( 0, end ):
#                A_i = curve.vertices[ i - 1 ]
#                B_i = curve.vertices[ i ]
#                if ( A_i == B_i ):
#                    continue
#                
#                for v in in_verts:
#                    # line equation
#                    # X = A + t * ( B - A )
#                    t = [ 0.0 for i in range(0,3) ]
#                    for j in range( 0, 3 ):
#                        if ( vertices[B_i][j] - vertices[A_i][j] != 0.0 ):
#                            t[j] = ( vertices[v][j] - vertices[A_i][j] ) / ( vertices[B_i][j] - vertices[A_i][j] )
#                    
#                    #point on line
#                    if ( t[0] == t[1] == t[2] ):
#                        #clip line
#                        #new lines ..-A-I and I-B-..
#                        if ( all( x > 0.0 and x < 1.0 for x in t ) ):
#                            curve.vertices = curve.vertices[i:]
#                            curve.vertices.insert( 0, v )
#                        
#                            temp_verts = temp_verts[:curve.vertices.index(A_i)]
#                            temp_verts.append( v )
#                    
#                            curves.append( Curve( temp_verts ) )
#                            temp_verts = curve.vertices.copy()

        
        print('Checking collisions')
        #LINE - LINE
        self.collision_line_line( scene )
        
        #LINE - TRIANGLE
        self.collision_line_triangle( scene )
        
        #TRIANGLE - TRIANGLE
        self.collision_triangle_triangle( scene )
        
        #self.check_curve_collision_with_itself( scene )
        
        print("--- %s seconds ---" % (time.time() - start_time))


        print('Build Octree')
        #Create octree
        x1 = y1 = z1 = -2 * scene.camera.data.clip_start
        x2 = y2 = z2 = scene.camera.data.clip_end
        
        tlf = [  x1,  y1, z1 ]
        brf = [ -x1, -y1, z1 ]
        tlb = [ -x2, -y2, z2 ]
        brb = [  x2,  y2, z2 ]
        
        root = Octree( tlf, brf, tlb, brb )
        
        #insert all objects to octree
        root.context = scene.curves + scene.polygons
        
        #divide tree
        root.divide_node( scene )
        
        print("--- %s seconds ---" % (time.time() - start_time))  

        self.filename = "file.svg"
        self.export_to_svg( scene, root )
        
        print('Finished')
        print("--- %s seconds ---" % (time.time() - start_time))
    
        return { 'FINISHED' }

                    
                        


    def compute_vp_matrix ( self, camera, render ):
        
        #compute view matrix
        v_mat = camera.matrix_world.inverted()
        
        #compute projection matrix
        p_mat = camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(),
            x = self.res[0],
            y = self.res[1],
            scale_x = render.pixel_aspect_x,
            scale_y = render.pixel_aspect_y )
        
        #update projection matrix
        #mirror in y-axis
        p_mat[1][1] *= -1.0
        #change z_NDC coords from (0,1) to (-1,1)
        n = camera.data.clip_start
        f = camera.data.clip_end
        p_mat[2][2] = ( 2*n + f ) / ( 2*n - f )
        p_mat[2][3] = ( 4*n*f )   / ( 2*n - f )
        
        return p_mat @ v_mat
    
    
    
    def compute_color ( self, diff_color, opac, cam_loc, light_dir, A, B, C, vp_mat_inv ):
        
        A = vp_mat_inv @ A.co
        B = vp_mat_inv @ B.co
        C = vp_mat_inv @ C.co
                        
        AB = ( B - A ).to_3d()
        AC = ( C - A ).to_3d()
        normal = AB.cross( AC )
        
        nl = max( normal.normalized().dot( light_dir ), 0.0 )
        amb_color = 0.2
        color = diff_color * amb_color + diff_color * nl
        color = Vector( ( round(c) for c in color ) ).to_4d()
        color[3] = opac
        
        return color

                        
                        
    def is_point_visible ( self, A ):
        
        for axis in range(3):
            if ( A[axis] < -A[3] or A[axis] > A[3] ):
                return False
            
        return True



    def verts_gen ( self, mesh, scene, mvp, v, first, triangle = None ):
    
        A = first
        if ( not triangle ):
            edges = ( ( edge.vertices[0], edge.vertices[1] ) for edge in mesh.edges )
        else:
            edges = ( ( triangle.vertices[tv - 1], triangle.vertices[tv] ) for tv in range(3) )
        
        
        for edge in edges:
            A_i = edge[0]
            B_i = edge[1]
            B = mvp @ mesh.vertices[ B_i ].co.to_4d()
                
            t_min = 0.0
            t_max = 1.0
            for sign in range(-1, 2 ,2):
                for axis in range(3):
                    M = -( sign * A[axis] + A[3] )
                    N = sign * ( B[axis] - A[axis] ) + B[3] - A[3]
                            
                    if ( N < 0.0 ):
                        t_max = min( t_max, M/N )
                    if ( N == 0.0 and M < 0.0 ):
                        continue
                    if ( N == 0.0 and M == 0.0 ):
                        continue
                    if ( N == 0.0 and M > 0.0 ):
                        t_min = 2.0
                    if ( N > 0.0 ):
                        t_min = max( t_min, M/N )
                            
                            
            #include only visible points
            if ( t_min < t_max ):    
                #new line is An..
                if ( t_min != 0.0 ):
                    An = A * ( 1.0 - t_min ) + B * t_min
                    key = 'I' + str( v + A_i ) + str( v + B_i )
                    scene.vertices[ key ] = Intersection( An, v + A_i, v + B_i )
                    yield key
                        
                #new line is ..Bn
                if ( t_max != 1.0 ):
                    Bn = A * ( 1.0 - t_max ) + B * t_max
                    key = 'I' + str( v + B_i ) + str( v + A_i )
                    scene.vertices[ key ] = Intersection( Bn, v + B_i, v + A_i )        
                else:
                    key = v + B_i
                    scene.vertices[ key ] = Vertex( B )
                yield key
                    
            A = B
        

    
    def curve_verts_gen ( self, mesh, scene, mvp, v ):
    
        #check first point
        A = mvp @ mesh.vertices[0].co.to_4d()
        if ( self.is_point_visible( A ) ):
            scene.vertices[v] = Vertex( A )
            yield v
                
        #check other points (edges)        
        for key in self.verts_gen( mesh, scene, mvp, v, A ):
            yield key


    def clip_curve ( self, obj, scene, vp_mat, obj_inds ):
        
        v = obj_inds['v']
        #get curve color
        try:
            color = [ round(x * 255) for x in obj.active_material.diffuse_color[:3] ]
            color.append( obj.active_material.diffuse_color[3] )
        except AttributeError:
            color = [ 0, 0, 0, 1.0 ]
            
        #compute MVP matrix
        mvp = vp_mat @ obj.matrix_world
            
        #tasselate current curve to lines
        mesh = obj.to_mesh()
            
        #curve processing
        begin = True
        for key in self.curve_verts_gen( mesh, scene, mvp, v ):
            if ( begin ):
                c = Curve( [key], color )
                scene.curves.append( c )
                begin = False
            else:
                c.vertices.append( key )
                if isinstance( scene.vertices[key], Intersection ):
                    begin = True
        
        obj_inds['v'] += len( mesh.vertices )
        obj.to_mesh_clear()
            
        

    def clip_triangles ( self, obj, scene, vp_mat, obj_inds ):
    
        v = obj_inds['v']
        p = obj_inds['p']
        #material base color
        if ( obj.material_slots ):
            mat = obj.material_slots[0].material
            try:
                principled_bsdf = next( x for x in mat.node_tree.nodes if x.type == 'BSDF_PRINCIPLED')
                base_color = principled_bsdf.inputs['Base Color']
                color = base_color.default_value
                diff_color = Vector( ( round(x * 255) for x in color[:3] ) )
                opacity = color[3]
            except:
                diff_color = Vector( ( 200, 200, 200 ) )
                opacity = 1.0
        else:
            diff_color = Vector( ( 200, 200, 200 ) )
            opacity = 1.0
            
        #compute MVP matrix
        mvp = vp_mat @ obj.matrix_world
            
        #tessellate current object to triangles
        mesh = obj.to_mesh()
        mesh_faces = mesh.calc_loop_triangles()
            
        #mesh processing
        for triangle in mesh.loop_triangles:
            #clip triangle
            temp_verts = []
            A = mvp @ mesh.vertices[ triangle.vertices[-1] ].co.to_4d()
            for key in self.verts_gen( mesh, scene, mvp, v, A, triangle ):
                temp_verts.append( key )
            
            
            #tri processing
            verts_c = len( temp_verts )
            if ( verts_c == 2 ):
                pass
                
            barren = []
            end = ( verts_c - 1 ) // 2
            for i in range( end ):
                for j in range( verts_c // 2 ):
                    l = j * 2
                    k = i + 1
                    new_tri_inds = [ temp_verts[l - 1*k], temp_verts[l], temp_verts[l + 1*k] ]
                        
                    #check if parallel lines
                    A = scene.vertices[ new_tri_inds[0] ]
                    B = scene.vertices[ new_tri_inds[1] ]
                    C = scene.vertices[ new_tri_inds[2] ]

                    AB = ( B - A )
                    AC = ( C - A )
                    diff = AB.normalized() - AC.normalized()
                    if ( abs( diff[0] ) < 0.00001 and abs( diff[1] ) < 0.00001 and abs( diff[2] ) < 0.00001 ):
                        new_tri_inds.sort( key = lambda x: ( scene.vertices[x] - A ) / AB )
                        #remove middle triangle vertex from polygon
                        barren.append( new_tri_inds[1] )
                        continue
                    
                    new_tri = Triangle( new_tri_inds, scene.vertices, len( scene.triangles ), p + triangle.polygon_index )
                    scene.triangles.append( new_tri )
                        
                    exists = next( ( x for x in scene.polygons if x.poly_index == p + triangle.polygon_index  ), None )                        
                    if ( not exists ):
                        #add vertices to polygon
                        poly_verts = [ el + v for el in mesh.polygons[ triangle.polygon_index ].vertices ]
                        #compute color of polygon
                        color = self.compute_color( diff_color, opacity, scene.camera_loc, scene.light_dir, A, B, C, vp_mat.inverted() )
                        #create new polygon and add it to scene
                        new_poly = Polygon( poly_verts, p + triangle.polygon_index, color, obj.name )
                        scene.polygons.append( new_poly )
                        exists = new_poly
                            
                    #add triangle to polygone
                    exists.tris.append( len( scene.triangles ) - 1 )
                            
                verts_c = 2
             
            #delete barren vertices from polygon
            if ( scene.polygons ):
                scene.polygons[-1].vertices = [ x for x in scene.polygons[-1].vertices if x not in barren ]
            
        temp_verts.clear()
        obj_inds['v'] += len( mesh.vertices )
        obj_inds['p'] += len( mesh.polygons )
        obj.to_mesh_clear()
                
                
                
    def clip_polygons ( self, scene ):
        
        for poly in reversed( scene.polygons ):
            intersections = [ tri_v for tri in poly.tris for tri_v in scene.triangles[tri].vertices if isinstance( scene.vertices[tri_v], Intersection ) ]
            #remove duplicates
            intersections = list( dict.fromkeys( intersections ) )
            
            #poly processing
            #Weiler-Atherton clipping algorithm
            if ( intersections ):                
                P, I, O = self.get_PIO( scene, poly, intersections )

                W = self.get_W( scene, I, O )
                
                temp_polys = self.WA_execute ( P, I, O, W )
                #not new polygones
                if ( len( temp_polys ) == 1 ):
                    poly.vertices = temp_polys[0].copy()
                #else split polygone
                else:
                    self.split_polygon( scene, poly, temp_polys )
                        
                temp_polys.clear()
                
                

    def find_cross_point_line_line ( self, A, B, C, D ):
        
        #determinant
        D_00 = B[0] - A[0]
        D_01 = D[0] - C[0]
        D_10 = B[1] - A[1]
        D_11 = D[1] - C[1]
        
        #right side
        b_0 = C[0] - A[0]
        b_1 = C[1] - A[1]
        
        try:
            det = D_00 * D_11 - D_10 * D_01
            t =  ( b_0*D_11 - b_1*D_01 ) / det
            s = -( D_00*b_1 - D_10*b_0 ) / det
        except ZeroDivisionError:
            return None
        
        if ( abs( t * ( B[2]-A[2] ) - s * ( D[2] - C[2] ) + ( A[2]-C[2] ) ) < 0.004 ):
            if( 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0 ):
                return C + ( D - C ) * s
            else:
                return None

        return None
    
    
    
    def collision_line_line ( self, scene ):
        
        #loop through curves except last
        f_end = len( scene.curves ) - 1
        #get first curve
        for f in range( f_end, 0, -1 ):
            inters = []
            first_c = scene.curves[ f ]

            #get second curve
            for s in range( f ):
                second_c = scene.curves[s]
                
                #get segment of first line
                for i in range( 1, len( first_c.vertices ) ):
                    A_i = first_c.vertices[ i - 1 ]
                    B_i = first_c.vertices[ i ]
                    A = scene.vertices[ A_i ]
                    B = scene.vertices[ B_i ]

                    #get segment of second line
                    for j in range( 1, len( second_c.vertices ) ):
                        C_i = second_c.vertices[ j - 1 ]
                        D_i = second_c.vertices[ j ]
                        C = scene.vertices[ C_i ]
                        D = scene.vertices[ D_i ]
                        
                        # find cross point, if exists
                        exists = self.find_cross_point_line_line( A, B, C, D )
                        if ( exists ):
                            key = 'I' + str(A_i) + str(B_i)
                            scene.vertices[ key ] = Intersection( exists, A_i, B_i )
                            inters.append( key )

            #curve processing
            for key in inters:
                ind = first_c.vertices.index( scene.vertices[key].sub_index )
                first_c.vertices.insert( ind, key )
                new = first_c.vertices[ :ind + 1 ]
                old = first_c.vertices[ ind: ]
                scene.curves.append( Curve( new, first_c.color ) )
                first_c.vertices = old
            inters.clear()

            
            
    def line_in_triangle ( self, scene, A, B, C, n, inter ):
                                
        #Barycentric coordinates
        u = (B - A).to_3d()
        v = (C - A).to_3d()
        w = (inter - A).to_3d()
                                
        gama = n.dot( u.cross( w ) ) / n.dot( n )
        beta = n.dot( w.cross( v ) ) / n.dot( n )
        alpha = 1.0 - gama - beta
                                
        #intersection with triangle
        if ( 0.00001 < alpha and alpha < 0.999 and 0.00001 < beta and beta < 0.999 and 0.00001 < gama and gama < 0.999 ):
            return True
        
        return False
            
    
    
    def find_cross_point_line_triangle ( self, scene, curve, triangle, start, end_loop ):
        
        n = triangle.normal.to_3d()        
        A = scene.vertices[ triangle.vertices[0] ].co                    
        d = n.dot( A )
        
        j = start
        #loop trhough all segments of line
        while( j < end_loop ):
            L_j = curve.vertices[ j - 1 ]
            M_j = curve.vertices[ j ]
#            if ( L_j == M_j ):
#                continue
            L = scene.vertices[ L_j ]
            M = scene.vertices[ M_j ]
                    
            f = n.dot( L.co )
            R = M - L
            e = n.dot( R )
                    
            if ( e == 0.0 ):
                j += 1
                continue
            x = ( d - f ) / e

            if ( x >= 0.00001 and x <= 0.999  ):
                #new vertex coordinates
                inter = L + (M - L) * x
                B = scene.vertices[ triangle.vertices[1] ].co
                C = scene.vertices[ triangle.vertices[2] ].co
                if ( self.line_in_triangle ( scene, A, B, C, n, inter ) ):
                    #tri-tri
                    if start == 0:
                        yield inter, L_j, M_j
                    #curve-tri
                    elif start == 1:
                        temp_verts = curve.vertices[:j]
                        curve.vertices = curve.vertices[j:]
                    
                        key = 'I' + str( L_j ) + str( M_j )
                        curve.vertices.insert( 0, key )
                        temp_verts.append( key )
                    
                        scene.curves.append( Curve( temp_verts, curve.color ) )
                    
                        inter = Intersection( inter, L_j, M_j )
                        scene.vertices[ key ] = inter
                    
                        end_loop = len( curve.vertices ) - 1
                        j = 0
                
            j += 1

            
            
    def collision_line_triangle ( self, scene ):

        #loop through curves in the scene
        for curve in reversed( scene.curves ):
            #loop through triangles in the scene
            for tri in scene.triangles:
                end_loop = len( curve.vertices ) - 1
                for _ in self.find_cross_point_line_triangle( scene, curve, tri, 1, end_loop ):
                    pass
                
                
                
    def collision_triangle_triangle ( self, scene ):
        
        zero = Vector( ( 0.0, 0.0, 0.0 ) )
        #loop through all polygons in the scene
        f_iter = 0
        f_end = len( scene.polygons ) - 1
        while ( f_iter < f_end ):
            f_intersections = []
            #loop through other polygons
            s_iter = f_iter + 1
            s_end  = f_end + 1
            for s_iter in range( s_iter, s_end ):
                f_poly = scene.polygons[f_iter]
                s_poly = scene.polygons[s_iter]
                #check if far enough                        
                if ( scene.overlap( f_poly, s_poly ) ):
                    #next polygon
                    continue
                
                s_intersections = []
                #loop through all triangles of polygon p
                f_tri_iter = 0
                f_tri_end  = len( f_poly.tris )
                while ( f_tri_iter < f_tri_end ):
                    tri_1 = scene.triangles[ f_poly.tris[ f_tri_iter ] ]
                    
                    n1 = tri_1.normal
                    
                    #loop through all triangles of polygon r
                    for s_tri in reversed( s_poly.tris ):
                        tri_2 = scene.triangles[ s_tri ]
                        
                        #check if parallel
                        n2 = tri_2.normal
                        cross = n1.cross(n2)
                        
                        if ( cross == zero ):
                            continue
                        
                        f_ints = []
                        for inter, A_i, B_i in self.find_cross_point_line_triangle( scene, tri_1, tri_2, 0, 3 ):
                            self.intersection_push_back( scene, f_iter, inter, A_i, B_i, f_ints )
                            
                        s_ints = []
                        for inter, A_i, B_i in self.find_cross_point_line_triangle( scene, tri_2, tri_1, 0, 3 ):
                            self.intersection_push_back( scene, s_iter, inter, A_i, B_i, s_ints )
                        
                        #compute next intersections in triangles
                        ints = f_ints + s_ints
                        f_stop = len(s_ints) + len(ints) % 2
                        s_stop = len(f_ints) + len(ints) % 2
                        
                        for inter, A_i, B_i in self.next_intersection( scene, tri_1, tri_2, ints, s_stop ):
                            self.intersection_push_back( scene, s_iter, inter, A_i, B_i, s_ints )
                            
                        for inter, A_i, B_i in self.next_intersection( scene, tri_2, tri_1, ints, f_stop ):
                            self.intersection_push_back( scene, f_iter, inter, A_i, B_i, f_ints )

                        #split triangles
                        if ( ints ):
                            #remove duplicates
                            f_ints = list( dict.fromkeys( f_ints ) )
                            
                            s_ints = list( dict.fromkeys( s_ints ) )
                            
                            #self.add_split_triangles( scene, tri_1, f_ints, f_iter )
                            if ( len( s_ints ) > 1 and len( f_ints ) > 1 ):
                                self.add_split_triangles( scene, tri_2, s_ints, s_iter )
                
                            s_intersections.extend( s_ints )
                            f_intersections.extend( f_ints )
                        s_ints.clear()
                        f_ints.clear()
                        
                    f_tri_iter += 1

                        
                #split second poly
                if (s_intersections):
                    self.polygon_slice( scene, s_iter, n1, s_intersections )

                    #remove duplicates
                    s_intersections = list( dict.fromkeys( s_intersections ) )
                    
                    #WA algorithm
                    P, I, O = self.get_PIO( scene, s_poly, s_intersections )
                    if ( not P ):
                        s_iter += 1
                        continue
                    
                    P.append( P[0] )
                    
                    I_temp = I.copy()
                    O = I[ 1::2 ]
                    I = I[ 0::2 ]
                    
                    if ( len(I) != len(O) ):
                        s_iter += 1
                        continue
                    
                    W = self.get_W_( scene, s_intersections, O )
                    
                    temp_polys = self.WA_execute ( P, I, O, W )
                    
                    I = I_temp[ 1::2 ]
                    O = I_temp[ 0::2 ]
                    W.reverse()

                    temp_polys.extend( self.WA_execute ( P, I, O, W ) )
                    
                    start = len(scene.polygons)
                    self.split_polygon( scene, s_poly, temp_polys )
#                    print('s poly   ', s_poly)
#                    print('new poly ', scene.polygons[-1])
#                    for t in scene.polygons[-1].tris:
#                        print( scene.triangles[t] )
                        
                    
                    
                    temp_polys.clear()
                    
                    
                    
                    
                if (f_intersections):
                    #scene.not_comp.append(f_poly)
                    continue
                    self.polygon_slice( scene, f_iter, n2, f_intersections )
                    
                    
                    #remove duplicates
                    f_intersections = list( dict.fromkeys( f_intersections ) )
                    
                    #WA algorithm
                    P, I, O = self.get_PIO( scene, f_poly, f_intersections )
                    if ( not P ):
                        scene.not_comp.append(f_poly)
                        f_iter += 1
                        continue
                    
                    P.append( P[0] )
                    
                    I_temp = I.copy()
                    O = I[ 1::2 ]
                    I = I[ 0::2 ]
                    
                    if ( len(I) != len(O) ):
                        scene.not_comp.append(f_poly)
                        f_iter += 1
                        continue
                    
                    W = self.get_W_( scene, f_intersections, O )
                    
                    temp_polys = self.WA_execute ( P, I, O, W )
                    
                    I = I_temp[ 1::2 ]
                    O = I_temp[ 0::2 ]
                    W.reverse()
                    
                    temp_polys.extend( self.WA_execute ( P, I, O, W ) )
                    
                    start = len(scene.polygons)
                    self.split_polygon( scene, f_poly, temp_polys )

                    
                    temp_polys.clear()

            
            f_end = len( scene.polygons ) - 1
            #next polygon
            f_iter += 1
        
        
        
    
    def intersection_push_back ( self, scene, poly_i, inter, A_i, B_i, ints ):
        
        poly = scene.polygons[ poly_i ]
        edges = [ [ poly.vertices[i-1], poly.vertices[i] ] for i in range( len( poly.vertices ) ) ]
        
        keys = next( ( edge for edge in edges if A_i in edge and B_i in edge ), None )
        if ( not keys ):
            exists = next( ( x for x in ints if scene.vertices[x].super_index == B_i and scene.vertices[x].sub_index == A_i ), None )
            if ( exists ):
                keys = [ B_i, A_i ]
            else:
                keys = [ A_i, B_i ]
                
        #error
        if ( len(keys) < 2 ):
            return
                            
        key = 'I' + str( keys[0] ) + str( keys[1] )
        inter = Intersection( inter, keys[0], keys[1] )
        scene.vertices[ key ] = inter
        ints.append( key )
        
        
        
    def next_intersection ( self, scene, f_tri, s_tri, ints, stop ):
        
        for i in range( stop ):
            subs = [ scene.vertices[int].sub_index for int in ints ]
                            
            N = f_tri.normal.normalized()
            d = N.dot( scene.vertices[ ints[0] ] )
                            
            verts = s_tri.vertices
            for edge in ( (verts[x-1], verts[x]) for x in range(3) if verts[x] not in subs ):
                f = N.dot( scene.vertices[ edge[0] ].co )
                R = scene.vertices[ edge[1] ].co - scene.vertices[ edge[0] ].co
                e = N.dot( R )
                if ( e == 0.0 ):
                    continue
                x = ( d - f ) / e    
                if ( 0.00001 <= x and x <= 0.9999 ):
                    yield scene.vertices[ edge[0] ].co + R * x, edge[0], edge[1]
                
                
            
            
            
    def polygon_slice ( self, scene, f_iter, n, f_ints ):
        
        poly = scene.polygons[f_iter]
        abc = f_ints.copy()
                            
        for i in abc:
            abcd = [ scene.vertices[i].super_index, scene.vertices[i].sub_index ]
            find = True
            while ( find ):
                find = False
                
                for tri_i in poly.tris:
                    ints = []
                    tri = scene.triangles[ tri_i ]
                    if ( not abcd ):
                        continue
                    
                    if ( all( x in tri.vertices for x in abcd ) ):                                            
                        find = True
                        N = n.normalized()
                        d = N.dot( scene.vertices[i] )
                        for j in range(3):
                            A_i = tri.vertices[j-1]
                            B_i = tri.vertices[j]
                            A = scene.vertices[A_i]
                            B = scene.vertices[B_i]
                            f = N.dot( A.co )
                            R = B.co - A.co
                            e = N.dot( R )
            
                            if ( e == 0.0 ):
                                continue
        
                            x = ( d - f ) / e
                            if ( 0.00001 <= x and x <= 0.9999 ):
                                #compute intersection
                                inter = A.co + R * x
                                
                                self.intersection_push_back ( scene, f_iter, inter, A_i, B_i, f_ints )
                            
                                ints.append( f_ints[-1] )
                        
                            
                        
                        #split triangle of second poly
                        if ( len( ints ) > 1 ):
                            new_tris = self.add_split_triangles( scene, tri, ints, f_iter )
                            x = next( ( x for x in ints if scene.vertices[x].super_index != abcd[0] or scene.vertices[x].sub_index != abcd[1] ), None )
                            abcd = [ scene.vertices[x].super_index, scene.vertices[x].sub_index ]
                        else:
                            abcd = None
                        ints.clear()
                        

                            
    
    
    def export_to_svg( self, scene, root ):

        try:    
            with open( self.filename ) as file:
                #File exists - delete content
                file = open( "./" + self.filename, "w" ).close()
                
        except FileNotFoundError:
            #File not exists - create new file
            file = open( "./" + self.filename, "w" ).close()
        except OSError as e:
            print(e)
            return ( 'CANCELLED' )
            
        file = open( "./" + self.filename, "a" )
        #append header
        file.write( f"""<svg xmlns="http://www.w3.org/2000/svg" """ + 
                    f"""width="{self.res[0]}" height="{self.res[1]}"> \n""" )
                    #f"""width="{self.res[0]}" height="{self.res[1]}" """ + 
                    #f"""shape-rendering="crispEdges"> \n""" )

        
        print('\nStart rendering')
        
        while ( root.childrens[0] or root.context ):
            root.traversal( file, scene, self.res )

            
            
            
        
        #append footer
        file.write( f"""</svg>""" )
        file.close()
    
        return { 'FINISHED' }



    
    def add_split_triangles ( self, scene, tri, ints, f_iter ):
        
        data = scene.vertices
        vertices = tri.vertices
        
        line_0 = [ ints[0], data[ ints[0] ].super_index, data[ ints[0] ].sub_index ]
        line_1 = [ ints[1], data[ ints[1] ].super_index, data[ ints[1] ].sub_index ]
        
        common_point = next( ( x for x in line_0 if x in line_1 ), None )
        
        line_0 = vertices.copy()
        line_0.remove( common_point )
        
        #remove triangle from polygon
        tri_i = scene.triangles.index( tri )
        poly = scene.polygons[ f_iter ]
        poly.tris.remove( tri_i )
        
        ind = len( scene.triangles )
        #add triangle
        vertices = [ common_point, ints[0], ints[1] ]
        scene.triangles.append( Triangle( vertices, scene.vertices, ind, tri.polygon ) )
        poly.tris.append( ind )
        
        ind += 1
        #add triangle
        vertices = [ ints[0], ints[1], line_0[0] ]
        scene.triangles.append( Triangle( vertices, scene.vertices, ind, tri.polygon ) )
        poly.tris.append( ind )
        
        ind += 1
        #add triangle
        index = next ( ( x for x in ints if data[x].super_index != line_0[0] and data[x].sub_index != line_0[0]  ), None )
        vertices = [ line_0[0], line_0[1], index ]
        scene.triangles.append( Triangle( vertices, scene.vertices, ind, tri.polygon ) )
        poly.tris.append( ind )
        
        
        
    def get_PIO ( self, scene, poly, intersections ):
        
        #vertices and intersections of polygon going in one direction
        P = poly.vertices.copy()
        #intersections where polygon lines enters clip space
        I = []
        #intersections where polygon lines exits clip space
        O = []
                    
        for v in range( len( poly.vertices ) ):
            #print('poly.vertices[i]', poly.vertices[i])
            for i in intersections:
                if ( scene.vertices[i].super_index == poly.vertices[v] and scene.vertices[i].sub_index == poly.vertices[v-1] ):
                    O.append( i )
                    ind = P.index( poly.vertices[v] )
                    P.insert( ind, i )
                            
                if ( scene.vertices[i].super_index == poly.vertices[v-1] and scene.vertices[i].sub_index == poly.vertices[v] ):
                    I.append( i )
                    ind = P.index( poly.vertices[v-1] ) + 1
                    P.insert( ind, i )
                
        if ( not I ):
            return None, None, None                
                
        #shift P
        while ( P[1] != I[0] ):
            P = P[1:] + P[:1]
            
        return P, I, O
    
    
    
    def get_W ( self, scene, I, O ):
        
        W12 = sorted( ( x for x in ( I + O ) if abs( -scene.vertices[x][0] - scene.vertices[x][3] ) <= 0.001 ), key = lambda x: scene.vertices[x][1] / scene.vertices[x][3] )
        W23 = sorted( ( x for x in ( I + O ) if abs( -scene.vertices[x][1] - scene.vertices[x][3] ) <= 0.001 ), key = lambda x: scene.vertices[x][0] / scene.vertices[x][3] )
        W34 = sorted( ( x for x in ( I + O ) if abs( scene.vertices[x][0] - scene.vertices[x][3] ) <= 0.001 ),  key = lambda x: scene.vertices[x][1] / scene.vertices[x][3] )
        W41 = sorted( ( x for x in ( I + O ) if abs( scene.vertices[x][1] - scene.vertices[x][3] ) <= 0.001 ),  key = lambda x: scene.vertices[x][0] / scene.vertices[x][3] )
        
        if ( W12 and W12[0] in I ):
            W12.reverse()
                            
        if ( W23 and W23[0] in I ):
            W23.reverse()
                            
        if ( W34 and W34[0] in I ):
            W34.reverse()
                            
        if ( W41 and W41[0] in I ):
            W41.reverse()
                            
        W = [0] + W12 + [1] + W23 + [2] + W34 + [3] + W41 + [0]
        
        return W
    
    
    
    def get_W_ ( self, scene, intersections, O ):
        
        intersections.sort( key = lambda x: scene.vertices[x][0] / scene.vertices[x][3] )
        intersections.sort( key = lambda x: scene.vertices[x][1] / scene.vertices[x][3] )
    
        W = intersections.copy()
        if ( W[0] not in O ):
            W.reverse()
            
        return W
        
    
    
    def WA_execute ( self, P, I, O, W ):
    
        #start
        i = P.index( I[0] )
        val = P[i]
        temp_poly = [val]
        temp_polys = []

        while ( True ):
            if ( val in I ):
                I.remove( val )
                i = P.index( val ) + 1
                val = P[i]
            elif ( val in O ):
                O.remove( val )
                i = W.index( val ) + 1
                
                if ( len(W) == i ):
                    #shift W
                    while ( W[0] != val ):
                        W = W[1:] + W[:1]
                i = W.index( val ) + 1
                
                val = W[i]
            else:
                if ( val in W ):
                    i = W.index( val ) + 1
                    if ( len(W) == i ):
                        #shift W
                        while ( W[0] != val ):
                            W = W[1:] + W[:1]
                    i = W.index( val ) + 1
                    val = W[i]
                else:
                    i = P.index( val ) + 1
                    val = P[i]
                                
            temp_poly.append( val )
            if ( temp_poly[0] == temp_poly[-1] ):
                temp_poly.remove( temp_poly[-1] )
                temp_polys.append( temp_poly.copy() )
                                
                if ( I ):
                    i = P.index( I[0] )
                    val = P[i]
                    temp_poly.clear()
                    temp_poly.append( val )
                else:
                    break
                
        return temp_polys
    
    
    
    def split_polygon ( self, scene, poly, temp_polys ):
        
        #add triangles to polygons
        temp_tris = poly.tris.copy()
                        
        #first poly
        poly.vertices = temp_polys[0].copy()
        for tri in temp_tris:
            res = all( x in poly.vertices for x in scene.triangles[ tri ].vertices )
            if ( not res ):
                poly.tris.remove( tri )
                    
        #other polys    
        for p_verts in temp_polys[1:]:
            p = Polygon( p_verts, len( scene.polygons ), poly.color, poly.id )
                    
            for tri in temp_tris:
                res = all( x in p_verts for x in scene.triangles[ tri ].vertices )
                if ( res ):
                    p.tris.append( tri )
                        
            scene.polygons.append( p )
            
            
            
    def check_curve_collision_with_itself ( self, scene ):
    
        #clip curve by colission with itself
        for curve in reversed( scene.curves ):
            i = len( curve.vertices ) - 2
            while ( i > 0 ):
                for j in range( i - 1, -1, -1 ):
                    #edge_i
                    A_i = curve.vertices[i + 1]
                    B_i = curve.vertices[i]
                    A = scene.vertices[A_i]
                    B = scene.vertices[B_i]
                    
                    A_n = ( A / A[3] ).to_2d()
                    B_n = ( B / B[3] ).to_2d()
                    
                    #edge_j
                    D_i = curve.vertices[j + 1]
                    E_i = curve.vertices[j]
                    D = scene.vertices[D_i]
                    E = scene.vertices[E_i]
                    
                    D_n = ( D / D[3] ).to_2d()
                    E_n = ( E / E[3] ).to_2d()
                    
                    inter_n = scene.find_line_line_intersection_2d( A_n, B_n, D_n, E_n )
                    if inter_n:
                        inter_f, inter_s = scene.get_intersections_l( inter_n, A, B, D, E )
                        #divide curve
                        if inter_f > inter_s:
                            scene.curves.append( Curve( curve.vertices[j+1:], curve.color ) )
                            curve.vertices = curve.vertices[:j+2]
                            i = len( curve.vertices ) - 2
                            j = 0
                            
                i -= 1
    







class Octree:
    
    def __init__ ( self, tlf, brf, tlb, brb, parent = None, depth = 0 ):
        #boundary points
        #top_left_front
        self.tlf = Vector( tlf )
        #bottom_right_front
        self.brf = Vector( brf )
        #top_left_back
        self.tlb = Vector( tlb )
        #bottom_right_back
        self.brb = Vector( brb )
        
        
        #compute center
        center_x = ( tlf[0] + brf[0] + tlb[0] + brb[0] ) / 4
        center_y = ( tlf[1] + brf[1] + tlb[1] + brb[1] ) / 4
        center_z = ( tlf[2] + tlb[2] ) / 2
        self.center = Vector( (center_x, center_y, center_z) )
        
        #compute boundary planes
        self.compute_planes()
        
        self.parent = parent
        self.childrens = [ None for i in range(8) ]
        
        self.context = list()
        
        self.depth = depth
        
    
    
    def compute_planes ( self ):
        
        #left plane
        
        A = self.tlf
        B = self.tlb
        C = Vector( ( self.tlf[0], 
                      self.brf[1], 
                      self.brf[2] ) )
        self.N_l = ( (C - A).cross( (B - A) ) ).normalized()
        
        #top plane
        C = Vector( ( self.brf[0], 
                      self.tlf[1], 
                      self.tlf[2] ) )
        self.N_t = ( (B - A).cross( (C - A) ) ).normalized()
            
        #right plane
        A = self.brf
        B = self.brb
        self.N_r = ( (C - A).cross( (B - A) ) ).normalized()
        
        #bottom plane
        C = Vector( ( self.tlf[0], 
                      self.brf[1], 
                      self.brf[2] ) )
        self.N_b = ( (B - A).cross( (C - A) ) ).normalized()
        
        
        
    def item_in_voxel ( self, scene, item ):

        res_l = []
        res_t = []
        res_r = []
        res_b = []
        res_front = []
        res_back  = []
        
        for v_i in item.vertices:
            V = scene.vertices[ v_i ].co.to_3d()
            X = V - self.tlf
            Y = V - self.brf
            res_l.append( self.N_l.dot( X ) )
            res_t.append( self.N_t.dot( X ) )
            res_r.append( self.N_r.dot( Y ) )
            res_b.append( self.N_b.dot( Y ) )
            if ( self.tlf[2] <= V[2] and V[2] <= self.tlb[2] ):
                res_front.append( 1.0 )
                res_back.append(  1.0 )
            else:
                res_front.append( -1.0 )
                res_back.append(  -1.0 )
            
        if  all( r >= 0.0 for r in res_l ) and \
            all( r >= 0.0 for r in res_t ) and \
            all( r >= 0.0 for r in res_r ) and \
            all( r >= 0.0 for r in res_b ) and \
            all( r >= 0.0 for r in res_front ) and \
            all( r >= 0.0 for r in res_back ):
            return True
                            
        return False
    
    
    
    def divide_voxel ( self ):
        
        tt = (self.tlb + self.tlf) / 2
        bb = (self.brf + self.brb) / 2
        
        tbf = (self.tlf + self.brf) / 2
        tbb = (self.tlb + self.brb) / 2
        
        depth = self.depth + 1
        
        #tlf, brf, tlb, brb
        self.childrens[0] = Octree( tt,
                                    self.center,
                                    self.tlb,
                                    tbb,
                                    self,
                                    depth )
        
        self.childrens[1] = Octree( Vector( ( self.center[0], tt[1], tt[2] ) ),
                                    Vector( ( bb[0] ,self.center[1], self.center[2] ) ),
                                    Vector( ( tbb[0], self.tlb[1], self.tlb[2] ) ),
                                    Vector( ( self.brb[0], tbb[1], self.brb[2] ) ),
                                    self,
                                    depth )
        
        self.childrens[2] = Octree( self.center,
                                    bb, 
                                    tbb,
                                    self.brb,
                                    self,
                                    depth )
        
        self.childrens[3] = Octree( Vector( ( tt[0], self.center[1], self.center[2] ) ),
                                    Vector( ( self.center[0], bb[1], self.center[2] ) ),
                                    Vector( ( self.tlb[0], tbb[1], self.tlb[2] ) ),
                                    Vector( ( tbb[0], self.brb[1], self.brb[2] ) ),
                                    self,
                                    depth )
        
        self.childrens[4] = Octree( self.tlf, 
                                    tbf, 
                                    tt,
                                    self.center,
                                    self,
                                    depth )
                                    
        self.childrens[5] = Octree( Vector( ( tbf[0], self.tlf[1], self.tlf[2] ) ), 
                                    Vector( ( self.brf[0], tbf[1], self.brf[2] ) ),
                                    Vector( ( self.center[0], tt[1], self.center[2] ) ),
                                    Vector( ( bb[0] ,self.center[1], self.center[2] ) ),
                                    self,
                                    depth )
        
        self.childrens[6] = Octree( tbf,
                                    self.brf,
                                    self.center,
                                    bb,
                                    self, 
                                    depth )
                                    
        self.childrens[7] = Octree( Vector( ( self.tlf[0], tbf[1], self.tlf[2] ) ),
                                    Vector( ( tbf[0], self.brf[1], self.brf[2] ) ),
                                    Vector( ( tt[0], self.center[1], self.center[2] ) ),
                                    Vector( ( self.center[0], bb[1], self.center[2] ) ),
                                    self,
                                    depth )



    def divide_node( self, scene ):
        
#        print('in root')
#        print(len( self.context ))
        num_of_items = 10000
        if ( len( self.context ) > num_of_items ):
            self.divide_voxel()
            
            #loop through children nodes
            for node in self.childrens:
                items_iter = 0
                items_end = len( self.context )
                #loop through context of parent node
                while( items_iter < items_end ):
                    item = self.context[ items_iter ]
                    if ( node.item_in_voxel( scene, item ) ):
                        node.context.append( item )
                        self.context.remove( item )
                        items_end = len( self.context )
                    else:
                        items_iter += 1
#                print('divide')
                node.divide_node( scene )
        
        #sort context after divide
#        print('\nunsorted context ', self.context)
        
        sorted_context = list()
        min = None
        i = 0
        while self.context:
            min = self.context[i]
            end = len( self.context )
            for j in range( 1, end ):
                s_item = self.context[j]
                #if min is above s_item
                is_above = scene.check_collision_clip_space( min, s_item )
                if not is_above:
                    pass
                    #s_item.elems_under.add( min )
                else:
                    #min.elems_under.add( s_item )
                    i = j
                    break
                
            else:
                sorted_context.append( min )
                self.context.remove( min )
                i = 0

        sorted_context.reverse()
        self.context = sorted_context.copy()
#        print('sorted context ', self.context)

        
        
    def traversal ( self, file, scene, res ):
        
        for elem in reversed( self.context ):
            for children in self.childrens:
                if children:
                    children.traversal( file, scene, res )
                
#            print('elem ', elem)
            #check collision with parent node
            if self.not_collision_with_parent( scene, elem ):
#                print('not collid with parent')
#                print('elems under ', elem.elems_under)
                #check collision with previous elems in self node
                elem_i = self.context.index( elem )
                for i in range( elem_i + 1, len( self.context ) ):
                    if self.context[i] in elem.elems_under:
#                        print('collision with self context')
                        break
                else:
#                    print('render ', elem, '\n')
                    print( len(self.context) )
                    elem.draw_object( file, scene, res )
                    self.context.remove( elem )
#            else:
#                print('collision with parent')
        
        x = 0
        for children in self.childrens:
            if children:
                x += children.traversal( file, scene, res )
            else:
                if not self.context:
                    return 1
        
        if x == 8:
            self.childrens = [ None for _ in range(8) ]
        
        return 0



    def not_collision_with_parent ( self, scene, f_elem ):

        if self.parent:
            if not self.parent.not_collision_with_parent( scene, f_elem ):
                return False
            
            for s_elem in self.parent.context:
                #if f_elem is above s_elem
                if scene.check_collision_clip_space( f_elem, s_elem ):
                    print(s_elem)
                    return False
                        
        return True
                
                



def register():
    bpy.utils.register_class(SVG_export)


def unregister():
    bpy.utils.unregister_class(SVG_export)

if __name__ == "__main__":
    register()
