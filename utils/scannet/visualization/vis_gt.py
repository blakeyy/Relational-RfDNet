'''
Visualization tools for Scannet.
author: ynie
date: July, 2020
'''
import sys
sys.path.append('.')
import os
from configs.path_config import PathConfig, ScanNet_OBJ_CLASS_IDS
import vtk
from utils.scannet.visualization.vis_scannet import Vis_Scannet
import numpy as np
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils.pc_util import rotz
import pickle
import random
import seaborn as sns
from utils.shapenet.common import Mesh

path_config = PathConfig('scannet')


class Vis_base(Vis_Scannet):
    '''
    visualization class for scannet frames.
    '''

    def __init__(self, scene_points, instance_models, center_list, vector_list, class_ids):
        self.scene_points = scene_points
        self.instance_models = instance_models
        self.cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        self.center_list = center_list
        self.vector_list = vector_list
        self.class_ids = class_ids
        self.palette_cls = np.array([*sns.color_palette("hls", len(ScanNet_OBJ_CLASS_IDS))])
        self.depth_palette = np.array(sns.color_palette("crest_r", n_colors=100))

    def set_ply_property(self, plyfile):

        plydata = vtk.vtkPLYReader()
        plydata.SetFileName(plyfile)
        plydata.Update()
        return plydata

    def set_arrow_actor(self, startpoint, vector):
        '''
        Design an actor to draw an arrow from startpoint to startpoint + vector.
        :param startpoint: 3D point
        :param vector: 3D vector
        :return: an vtk arrow actor
        '''
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.2)
        arrow_source.SetTipRadius(0.08)
        arrow_source.SetShaftRadius(0.02)

        vector = vector / np.linalg.norm(vector) * 0.5

        endpoint = startpoint + vector

        # compute a basis
        normalisedX = [0 for i in range(3)]
        normalisedY = [0 for i in range(3)]
        normalisedZ = [0 for i in range(3)]

        # the X axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(endpoint, startpoint, normalisedX)
        length = math.Norm(normalisedX)
        math.Normalize(normalisedX)

        # the Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalisedX, arbitrary, normalisedZ)
        math.Normalize(normalisedZ)

        # the Y axis is Z cross X
        math.Cross(normalisedZ, normalisedX, normalisedY)

        # create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalisedX[i])
            matrix.SetElement(i, 1, normalisedY[i])
            matrix.SetElement(i, 2, normalisedZ[i])

        # apply the transform
        transform = vtk.vtkTransform()
        transform.Translate(startpoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # create a mapper and an actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()

        mapper.SetInputConnection(arrow_source.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)

        return actor

    def set_bbox_line_actor(self, corners, faces, color):
        edge_set1 = np.vstack([np.array(faces)[:, 0], np.array(faces)[:, 1]]).T
        edge_set2 = np.vstack([np.array(faces)[:, 1], np.array(faces)[:, 2]]).T
        edge_set3 = np.vstack([np.array(faces)[:, 2], np.array(faces)[:, 3]]).T
        edge_set4 = np.vstack([np.array(faces)[:, 3], np.array(faces)[:, 0]]).T
        edges = np.vstack([edge_set1, edge_set2, edge_set3, edge_set4])
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        pts = vtk.vtkPoints()
        for corner in corners:
            pts.InsertNextPoint(corner)

        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        for edge in edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
            colors.InsertNextTuple3(*color)

        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(pts)
        linesPolyData.SetLines(lines)
        linesPolyData.GetCellData().SetScalars(colors)

        return linesPolyData

    def get_bbox_line_actor(self, center, vectors, color, opacity, width=10):
        corners, faces = self.get_box_corners(center, vectors)
        bbox_actor = self.set_actor(self.set_mapper(self.set_bbox_line_actor(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        bbox_actor.GetProperty().SetLineWidth(width)
        return bbox_actor

    def set_render(self, centroid, only_points, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw world system'''
        # renderer.AddActor(self.set_axes_actor())

        '''set camera'''
        camera = self.set_camera(centroid, [[0., 0., 0.], [-centroid[0], -centroid[1], centroid[0]**2/centroid[2] + centroid[1]**2/centroid[2]]], self.cam_K)
        renderer.SetActiveCamera(camera)

        '''draw scene points'''
        colors = np.linalg.norm(self.scene_points[:, :3]-centroid, axis=1)
        colors = self.depth_palette[np.int16((colors-colors.min())/(colors.max()-colors.min())*99)]
        point_actor = self.set_actor(
            self.set_mapper(self.set_points_property(self.scene_points[:, :3], 255*colors), 'box'))
        point_actor.GetProperty().SetPointSize(3)
        point_actor.GetProperty().SetOpacity(0.3)
        point_actor.GetProperty().SetInterpolationToPBR()
        renderer.AddActor(point_actor)

        if not only_points:
            '''draw shapenet models'''
            palette_inst = np.array([*sns.color_palette("hls", 20)])

            dists = np.linalg.norm(np.array(self.center_list) - centroid, axis=1)
            min_max_dist = [min(dists), max(dists)]
            dists = (dists - min_max_dist[0])/(min_max_dist[1]-min_max_dist[0])
            dists = np.clip(dists, 0, 1)
            inst_color_ids = np.round(dists*(palette_inst.shape[0]-1)).astype(np.uint8)

            for obj, cls_id, color_id in zip(self.instance_models, self.class_ids, inst_color_ids):
                object_actor = self.set_actor(self.set_mapper(obj, 'model'))
                object_actor.GetProperty().SetColor(self.palette_cls[cls_id])
                object_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(object_actor)

            '''draw bounding boxes'''
            for center, vectors, cls_id, color_id in zip(self.center_list, self.vector_list, self.class_ids, inst_color_ids):
                # box_line_actor = self.get_bbox_line_actor(center, vectors, 255*self.palette_cls[cls_id], 1., 10)
                # box_line_actor.GetProperty().SetInterpolationToPBR()
                # renderer.AddActor(box_line_actor)

                corners, faces = self.get_box_corners(center, vectors)
                bbox_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, 255*self.palette_cls[cls_id]), 'box'))
                bbox_actor.GetProperty().SetOpacity(0.2)
                bbox_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(bbox_actor)

                # draw orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(center, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, -10, 10), (-10, -10, 10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1.5)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer

    def set_render_window(self, centroid, only_points):
        render_window = vtk.vtkRenderWindow()
        renderer = self.set_render(centroid, only_points)
        renderer.SetUseDepthPeeling(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(*np.int32((self.cam_K[:2,2]*2)))

        return render_window

    def visualize(self, centroid=np.array([0, -2.5, 2.5]), save_path = None, only_points=False):
        '''
        Visualize a 3D scene.
        '''

        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window = self.set_render_window(centroid, only_points)
        render_window_interactor.SetRenderWindow(render_window)
        render_window.Render()

        if save_path is not None:
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(render_window)
            windowToImageFilter.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(save_path)
            writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            writer.Write()

        render_window_interactor.Start()

if __name__ == '__main__':
    scene_dirname = 'scene0704_01'
    #scene_dirname = 'scene0466_01'
    #scene_dirname = 'scene0359_01'
    #scene_dirname = 'scene0317_01'
    fused_points = np.load(os.path.join(path_config.processed_data_path, scene_dirname, 'full_scan.npz'))['mesh_vertices'][:,:3]
    with open(os.path.join(path_config.processed_data_path, scene_dirname, 'bbox.pkl'), 'rb') as file:
        bboxes = pickle.load((file))

    FLAP_YZ = False
    FLAP_XZ = False
    AUGMENT_ROT = False

    if FLAP_YZ:
        # Flipping along the YZ plane
        fused_points[:, 0] = -1 * fused_points[:, 0]
    if FLAP_XZ:
        # Flipping along the XZ plane
        fused_points[:, 1] = -1 * fused_points[:, 1]

    if AUGMENT_ROT:
        rot_angle = (np.random.random() * np.pi / 2) - np.pi / 4
    else:
        rot_angle = 0.
    rot_mat = rotz(rot_angle)
    fused_points[:, 0:3] = np.dot(fused_points[:, 0:3], np.transpose(rot_mat))

    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    instance_models = []
    center_list = []
    vector_list = []
    class_ids = []
    for box in bboxes:
        if FLAP_YZ:
            box['box3D'][0] = -1 * box['box3D'][0]
            box['box3D'][6] = np.sign(box['box3D'][6]) * np.pi - box['box3D'][6]
        if FLAP_XZ:
            box['box3D'][1] = -1 * box['box3D'][1]
            box['box3D'][6] = -1 * box['box3D'][6]

        box['box3D'][0:3] = np.dot(box['box3D'][0:3], np.transpose(rot_mat))
        box['box3D'][6] += rot_angle

        '''Normalize angles to [-np.pi, np.pi]'''
        box['box3D'][6] = np.mod(box['box3D'][6] + np.pi, 2 * np.pi) - np.pi

        shapenet_model = os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path, box['shapenet_catid'], box['shapenet_id'] + '.off')
        assert os.path.exists(shapenet_model)

        obj_model = os.path.join('./temp', box['shapenet_catid'], box['shapenet_id'] + '.obj')
        if not os.path.exists(obj_model):
            if not os.path.exists(os.path.dirname(obj_model)):
                os.mkdir(os.path.dirname(obj_model))
            Mesh.from_off(shapenet_model).to_obj(obj_model)
        vtk_object = vtk.vtkOBJReader()
        vtk_object.SetFileName(obj_model)
        vtk_object.Update()
        # get points from object
        polydata = vtk_object.GetOutput()
        # read points using vtk_to_numpy
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)

        '''Fit obj points to bbox'''
        obj_points = obj_points - (obj_points.max(0) + obj_points.min(0))/2.
        obj_points = obj_points.dot(transform_m.T)
        obj_points = obj_points.dot(np.diag(1/(obj_points.max(0) - obj_points.min(0)))).dot(np.diag(box['box3D'][3:6]))
        orientation = box['box3D'][6]
        axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
        obj_points = obj_points.dot(axis_rectified) + box['box3D'][0:3]

        points_array = numpy_to_vtk(obj_points[..., :3], deep=True)
        polydata.GetPoints().SetData(points_array)
        vtk_object.Update()

        '''draw bboxes'''
        center = box['box3D'][:3]
        orientation = box['box3D'][6]
        axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
        vectors = np.diag(box['box3D'][3:6]/2.).dot(axis_rectified)

        instance_models.append(vtk_object)
        center_list.append(center)
        vector_list.append(vectors)
        class_ids.append(list(ScanNet_OBJ_CLASS_IDS).index(box['cls_id']))

    scene = Vis_base(scene_points=fused_points, instance_models=instance_models, center_list=center_list,
                     vector_list=vector_list, class_ids=class_ids)
    save_path = 'out/samples'
    save_path = os.path.join(save_path, scene_dirname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    scene.visualize(centroid=np.array([3, 0, 3]), save_path=os.path.join(save_path, 'verify.png'))
