import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np
import SimpleITK as sitk
import shutil
import sitkUtils
#
# spline
#

class spline(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "spline"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Mehran Azimbagirad, Valerie Burdin, Guillaume Dardenne (Latim Lab)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#spline">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
Mehran Azimbagirad, Valerie Burdin, Guillaume Dardenne (Latim Lab).This work has benefited from a French grant managed by the National Research Agency under the “programme dinvestissements davenir” bearing the reference ANR-17- RHUS-0005, and the financial support of the Brittany Region
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # spline1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='spline',
    sampleName='spline1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'spline1.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='spline1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='spline1'
  )

  # spline2
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='spline',
    sampleName='spline2',
    thumbnailFileName=os.path.join(iconsPath, 'spline2.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='spline2.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='spline2'
  )

#
# splineWidget
#

class splineWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/spline.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    #print(dir(self.ui.tableWidget))
    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = splineLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.num_label_SliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.smoothingctkSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    #self.ui.tableWidget.connect("valueChanged(QTableWidget)", self.updateParameterNodeFromGUI)
    self.ui.interp_dirctkComboBox.connect("currentTextChanged(QString)", self.updateParameterNodeFromGUI)
    #self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True
    #print(dir(self._parameterNode))
    # Update node selectors and sliders
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.outputSelector2.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume2"))
    #self.ui.tableWidget.setCurrentItem(self._parameterNode.GetNodeReference("QTable"))
    self.ui.num_label_SliderWidget.value = float(self._parameterNode.GetParameter("num_label"))
    self.ui.smoothingctkSliderWidget.value = float(self._parameterNode.GetParameter("sigma"))
    #self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")
    self.ui.interp_dirctkComboBox.currentText = str(self._parameterNode.GetParameter("interp_dirLabel"))

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Compute output volume"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"
      self.ui.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
    #print(dir(self.ui.tableWidget))
    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume2", self.ui.outputSelector2.currentNodeID)
    self._parameterNode.SetParameter("num_label", str(self.ui.num_label_SliderWidget.value))
    self._parameterNode.SetParameter("sigma", str(self.ui.smoothingctkSliderWidget.value))
    #self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
    self._parameterNode.SetParameter("interp_dirLabel", str(self.ui.interp_dirctkComboBox.currentText))
    #self._parameterNode.SetcurrentItem("QTable",self.ui.tableWidget.currentItem)

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:

      # Compute output
      self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
        self.ui.num_label_SliderWidget.value, self.ui.outputSelector2.currentNode(),self.ui.smoothingctkSliderWidget.value,   self.ui.interp_dirctkComboBox.currentText) #, self.ui.invertOutputCheckBox.checked ,self.ui.tableWidget.currentItem

      # Compute inverted output (if needed)
      #if self.ui.invertedOutputSelector.currentNode():
        # If additional output volume is selected then result with inverted num_label is written there
       # self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
        #  self.ui.num_label_SliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()

    
#
# splineLogic
#

class splineLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("num_label"):
      parameterNode.SetParameter("num_label", "1")
    if not parameterNode.GetParameter("sigma"):
      parameterNode.SetParameter("sigma", "1.0")
    #if not parameterNode.GetParameter("Invert"):
     # parameterNode.SetParameter("Invert", "false")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("interp_dirLabel", "Axial")

  def process(self, inputVolume, outputVolume, num_label, outputVolume2, sigma, interp_dirLabel, showResult=True):#invert=False QTable,
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be num_labeled
    :param outputVolume: num_labeling result
    :param imagenum_label: values above/below this num_label will be set to 0
    :param invert: if True then values above the num_label will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """
    #print(QTable)
    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")
    import time
    startTime = time.time()
    logging.info('Processing started')

    # Compute the num_labeled output volume using the "num_label Scalar Volume" CLI module
#    cliParams = {
#      'InputVolume': inputVolume.GetID(),
#      'OutputVolume': outputVolume.GetID(),
#      'num_labelValue' : imagenum_label,
#      'num_labelType' : 'Above' if invert else 'Below'
#      }
#    cliNode = slicer.cli.run(slicer.modules.num_labelscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
#********************** My Code ***********************************************************    
    #inputVolume
    print('Author:Mehran Azimbagirad')
    #print('all variables',locals())
    #print(inputVolume.GetID())
    #print('input volume dir=',dir(inputVolume))
    
    #print('my_fun=',dir(my_fun))
    #my_fun.contours(25)
    #contours(25)
    
    #********************* Step 1: extract labels ******************************
    try:
        shutil.rmtree('temp')
        shutil.rmtree('temp')
    except:
        os.mkdir('temp')
    os.mkdir('temp/1label')
    progressbar = slicer.util.createProgressDialog(autoClose=True)
    progressbar.value = 5
    progressbar.labelText = "processing started ..."
    slice_ex(inputVolume,num_label,'temp/1label',interp_dirLabel)
    progressbar.value = 10
    #********************* Step 2: create contours *****************************
    progressbar.labelText = "creating contours ..."
    os.mkdir('temp/2contours')
    contours('temp/1label','temp/2contours',num_label,interp_dirLabel)
    progressbar.value = 20
        #************* Registration of the contours **************
    progressbar.labelText = "registration contours ..."
    reg_cont('temp/2contours',num_label)
    progressbar.value = 30
        #************* Correspondent points **********************
    progressbar.labelText = "corresponding points ..."
    slicer.util.pip_install("networkx")
    slicer.util.pip_install("scipy")
    slicer.util.pip_install("multiprocess")
    corr_points('temp/2contours',num_label)
    progressbar.value = 80
    #************** Fill contours **********************************************
    os.mkdir('temp/3spline_interpolate')
    fill_contours('temp/1label','temp/2contours',num_label,'temp/3spline_interpolate',interp_dirLabel)
    progressbar.value = 90
    #************** smoothing labels********************************************
    progressbar.labelText = "finalizing ..."
    os.mkdir('temp/4all_labels')
    smooth_label('temp/3spline_interpolate',num_label,sigma,'temp/4all_labels')
    #************** add labels **************************************************
    all_labels('temp/4all_labels',num_label,'temp')
    #******************** set outputs *********************************************
    name2=outputVolume2.GetName()
    slicer.mrmlScene.RemoveNode(outputVolume2)
    outputVolume2=slicer.util.loadLabelVolume('temp/Intersection.nii.gz', properties={}, returnNode=False)
    outputVolume2.SetName(name2)
    
    name1=outputVolume.GetName()
    slicer.mrmlScene.RemoveNode(outputVolume)
    outputVolume=slicer.util.loadLabelVolume('temp/all_labels.nii.gz', properties={}, returnNode=False)
    outputVolume.SetName(name1)
    
    progressbar.value = 100
#******************************************************************************************
    #slicer.mrmlScene.RemoveNode(inputVolume)
    #slicer.util.setSliceViewerLayers(background=slicer.util.getNode(inputVolume.GetName()), foreground='keep-current', label=slicer.util.getNode(outputVolume.GetName()), foregroundOpacity=None, labelOpacity=0.5, fit=False)
    slicer.util.setSliceViewerLayers(background='keep-current', foreground=slicer.util.getNode(inputVolume.GetName()), label=slicer.util.getNode(outputVolume.GetName()), foregroundOpacity=0.5, labelOpacity=0.5, fit=False)

    stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

#
# splineTest
#

class splineTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_spline1()

  def test_spline1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """
    self.delayDisplay("Starting the test")
    # Get/create input data
    #import SampleData
    #registerSampleData()
    #inputVolume = SampleData.downloadSample('spline1')
    #self.delayDisplay('Loaded test data set')

    #inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    #self.assertEqual(inputScalarRange[0], 0)
    #self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    num_label = 1

    # Test the module logic

    logic = splineLogic()

    # Test algorithm with non-inverted num_label
    #logic.process(inputVolume, outputVolume, num_label, True)
    #outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    #self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    #self.assertEqual(outputScalarRange[1], num_label)

    # Test algorithm with inverted num_label
    #logic.process(inputVolume, outputVolume, num_label, False)
    #outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    #self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    #self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')
#********************************************************************************************
#************************* My super functions *****************************************************
#**************************************************************************************************
#***************************** extract volume labels **********************************************
def slice_ex(input_label,numberof_labels,output_folder,interp_dirLabel):
    #print('dir of inputVolume.GetImageData()=',dir(input_label.GetImageData()))
    label = input_label.GetImageData()
    #label=sitk.ReadImage(input_label)
    #label=slicer.util.addVolumeFromArray(input_label.GetID()) 
    #print(type(label))   
    dirs = np.zeros([3,3])
    input_label.GetIJKToRASDirections(dirs)
    dirs=np.multiply(dirs,np.array([[-1,-1,-1],[-1,-1,-1],[1,1,1]]))
    direction=np.reshape(dirs,(1,9)).tolist()
    #print('input_label.GetDirection()======',direction[0])
    #print('type(label)=',type(label))
    origin=input_label.GetOrigin()
    #print(origin[0], type(origin))
    labelArray =slicer.util.array(input_label.GetID())
    
    #dirs1 = vtk.vtkMatrix4x4()
    #input_label.GetIJKToRASDirectionMatrix(dirs1)
    #print('input_label.GetDirection()======',dirs1)
    #*****************label********************************    
    #resultImage = sitk.GetImageFromArray(labelArray)
    #resultImage.CopyInformation(label)

    #print('current folder=',os.getcwd())
    #resultImage.SetOrigin(np.int(label.GetOrigin()))
    #l_name=outputDir+'all_label_.nii.gz'
    #sitk.WriteImage(resultImage, l_name)
    labels=np.arange(numberof_labels)
    for l in labels:
        label_i=np.copy(labelArray)
        label_i[label_i!=l+1]=0
        resultImage = sitk.GetImageFromArray(label_i)
        #print('sitk image dir==',dir(resultImage))
        resultImage.SetSpacing(input_label.GetSpacing())
        #print('direction=',direction[0])
        resultImage.SetOrigin((origin[0]*-1.0, origin[1]*-1.0,origin[2]) )
        resultImage.SetDirection(direction[0])
        #resultImage.CopyInformation(label.GetInformation())
        #print('new direction=',resultImage.GetDirection())
        l_name=output_folder+'/label_'+str(int(l+1))+'.nii.gz'
        print(l_name)
        sitk.WriteImage(resultImage, l_name)
    return 1
#**************************************************************************************************
#******************************** Contours *********************************************
#**************************************************************************************************
def contours(input_folder,output_folder,numberof_labels,interp_dirLabel):
    #for L in range(int(numberof_labels)):
    #labels_number=np.arange(numberof_labels)
    #folders = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
    #c=0
    #for file_name in folders:
    for L in range(int(numberof_labels)):
        #c=c+1
        os.mkdir(output_folder+'/label_'+str(int(L+1)))
        outputDir = output_folder+'/label_'+str(int(L+1))
        f_name=input_folder+'/label_'+str(int(L+1))+'.nii.gz'
        label = sitk.ReadImage(f_name,sitk.sitkInt16)
        labelArray0 = sitk.GetArrayFromImage(label)
        labelArray0[labelArray0.nonzero()]=1
        centroid=[]
#******************* If direction is not right, we need to rotate it****************
        if interp_dirLabel=='Axial':
            labelArray=np.zeros((labelArray0.shape[2], labelArray0.shape[0], labelArray0.shape[1]))
            for i in range(labelArray0.shape[0]):
                #labelArray[:,i,:]=labelArray0[i,:,:]
                labelArray[:,i,:]=np.transpose(labelArray0[i,:,:])
        elif interp_dirLabel=='Sagittal':
            labelArray=np.zeros((labelArray0.shape[1], labelArray0.shape[2], labelArray0.shape[0]))
            for i in range(labelArray0.shape[2]):
                #labelArray[:,i,:]=labelArray0[i,:,:]
                labelArray[:,i,:]=np.transpose(labelArray0[:,:,i])
        else :#interp_dirLabel=='Coronal'
            labelArray=np.zeros((labelArray0.shape[2], labelArray0.shape[1], labelArray0.shape[0]))
            for i in range(labelArray0.shape[1]):
                #labelArray[:,i,:]=labelArray0[i,:,:]
                labelArray[:,i,:]=np.transpose(labelArray0[:,i,:])

#target_points=[]
#c=0
        for i in range(labelArray.shape[1]):
            if (np.sum(labelArray[:,i,:])>=1):    
        #********** find the centroid ********        
        #**********method 1 
                centroid_index = np.floor([nz.mean() for nz in labelArray[:,i,:].nonzero()])
                centroid.append([int(centroid_index[0]),i,int(centroid_index[1])])
        #***********method 2:
        if len(centroid)>50:
            raise ValueError("Too much slices!!!Probably wrong direction was selected")
        #***********************************************************************************
        file_name=outputDir+'/centroid_ponits.txt'
        out = open(file_name, 'w')
        for s in centroid:
            point=str(s[0])+','+str(s[1])+','+str(s[2])+'\n'  
            out.write(point)
        out.close()
#***************** OutPut the correspondent points for interpolation*************************
        borders_with_angels=[]
#**************** degree assignment ***************
#epsilon=0.01
        current_border_angle=[]
        for j in range(len(centroid)):#range(len(centroid)): 11 slices
            borders=GetBorders(labelArray[:,centroid[j][1],:],centroid[j][1])
            current_border_angle.clear()
            for k in range(len(borders)):
                angle=get_angle(centroid[j][0],centroid[j][2],borders[k][0],borders[k][2])*180/np.pi
                dist_to_center=np.sqrt((borders[k][0]-centroid[j][0])*(borders[k][0]-centroid[j][0])+(borders[k][2]-centroid[j][2])*(borders[k][2]- centroid[j][2]))
                current_border_angle.append([borders[k],angle,dist_to_center])  
            borders_with_angels.append(current_border_angle[:])
#************************************** sort contours points **********************************
        for j in range(len(centroid)):
            last_ind=len(borders_with_angels[j])    
            contour1=[]
            contour1.append(borders_with_angels[j][0][:])
            borders_with_angels[j].remove(contour1[0])
            i=1
            while i <last_ind:
                min_ind, value=min_dist(borders_with_angels[j],contour1[-1][0])
                if value<=8:
                    min_ind.sort()
                    for k in min_ind:
                        contour1.append(borders_with_angels[j][k][:])
                    min_ind.sort(reverse=True)
                    for k in min_ind:
                        borders_with_angels[j].remove(borders_with_angels[j][k][:]) 
                    i=i+len(min_ind)
                else:
                    i=i+1
            file_name=outputDir+'/contour_points_'+str(j+1)+'.txt'
            out = open(file_name, 'w')
            for s in range(len(contour1)):
                point=str(contour1[s][0][0])+','+str(contour1[s][0][1])+','+str(contour1[s][0][2])+','+str(contour1[s][1])+','+str(contour1[s][2])  +'\n'  
                out.write(point)
            out.close()
        #---------------------------- Make 2D image ---------------------------------------------
        cc=1
        for i in range(len(centroid)):
            label_new=labelArray[:,centroid[i][1],:]
            resultImage = sitk.GetImageFromArray(label_new)
            resultImage.SetOrigin(label.GetOrigin())
            fname=outputDir+'/Slice_'+str(cc)+'.nii.gz'
            sitk.WriteImage(resultImage, fname)
            cc=cc+1
        #elapsed_time_fl = (time.time() - start)
        #print('CPU time=',elapsed_time_fl)
#**************************************************************************************************
#*********************** Registration contours ***************************************************
#**************************************************************************************************
def reg_cont(input_folder,numberof_labels):
    for L in range(int(numberof_labels)):
        inputDir1 = input_folder+'/label_'+str(L+1) #'input folder'
        centroids=np.loadtxt(inputDir1+'/centroid_ponits.txt',delimiter=',')
        contours=[]
        for i in range(centroids.shape[0]):
            fname=inputDir1+'/contour_points_'+str(i+1)+'.txt'
            c1=np.loadtxt(fname,delimiter=',')
            c2=c1.tolist()
            contours.append(c2[:])
        R = sitk.ImageRegistrationMethod()
#------------- Similarity metrics-------------------------
        R.SetMetricAsMeanSquares()
#------------- Optimizer -----------------------------------
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5,minStep=1e-6,numberOfIterations=200,gradientMagnitudeTolerance=1e-8)
#-------------- Interpolator --------------------------------
        R.SetInterpolator(sitk.sitkLinear)
        Rotations=[]
        for i in range(centroids.shape[0]-1):
            image1=sitk.ReadImage(inputDir1+'/Slice_'+str(i+1)+'.nii.gz',sitk.sitkFloat32)
            image2=sitk.ReadImage(inputDir1+'/Slice_'+str(i+2)+'.nii.gz',sitk.sitkFloat32)
            if len(contours[i+1])>=len(contours[i]):
                moving=image1
                fixed =image2
                initialTx = sitk.CenteredTransformInitializer(fixed, moving,sitk.Similarity2DTransform())
                R.SetInitialTransform(initialTx)
                R.DebugOff()
                outTx = R.Execute(fixed, moving)
                #---------- Transform contour points ------------------------------
                y=[item[0] for item in contours[i]]
                x=[item[2] for item in contours[i]]
                #scale,rotate,translationx,translationy=outTx.GetInverse().GetParameters()
                scale,rotate,translationx,translationy=outTx.GetParameters()
                Centerx, Centery=outTx.GetFixedParameters()
                Rotations.append(rotate*180/np.pi)
                My_transform=sitk.Transform(sitk.Similarity2DTransform())
                My_transform.SetParameters([scale,rotate,translationx,translationy])
                My_transform.SetFixedParameters([Centerx, Centery])
      
                Xt=[]
                Yt=[]
                ind2physX=[]
                ind2physY=[]
                for j in range(len(x)):
                    ind2physx,ind2physy=image1.TransformIndexToPhysicalPoint((int(x[j]),int(y[j])))
                    ind2physX.append(ind2physx)
                    ind2physY.append(ind2physy)
                for j in range(len(x)):
                    xt,yt=outTx.GetInverse().TransformPoint((ind2physX[j],ind2physY[j]))
                    ind2physx,ind2physy=image2.TransformPhysicalPointToIndex((xt,yt))
                    Xt.append(ind2physx)
                    Yt.append(ind2physy)
                y0=[item[0] for item in contours[i+1]]
                x0=[item[2] for item in contours[i+1]]
                center_transformed=centroids[i+1,1]
            else:
                print('inverse registration')
                moving=image2
                fixed =image1
                initialTx = sitk.CenteredTransformInitializer(fixed, moving,sitk.Similarity2DTransform())
                #initialTx = sitk.CenteredTransformInitializer(fixed, moving,sitk.AffineTransform(fixed.GetDimension()))
                #initialTx = sitk.Similarity2DTransform()
                R.SetInitialTransform(initialTx)
                R.DebugOff()
                outTx = R.Execute(fixed, moving)
                #scale,rotate,translationx,translationy=outTx.GetInverse().GetParameters()
                scale,rotate,translationx,translationy=outTx.GetParameters()
                Centerx, Centery=outTx.GetFixedParameters()
                Rotations.append(rotate*180/np.pi)

                My_transform=sitk.Transform(sitk.Similarity2DTransform())
                My_transform.SetParameters([scale,0,translationx,translationy])
                My_transform.SetFixedParameters([Centerx, Centery])
                
                y=[item[0] for item in contours[i+1]]
                x=[item[2] for item in contours[i+1]]
                Xt=[]
                Yt=[]
                ind2physX=[]
                ind2physY=[]
                for j in range(len(x)):
                    ind2physx,ind2physy=image2.TransformIndexToPhysicalPoint((int(x[j]),int(y[j])))
                    ind2physX.append(ind2physx)
                    ind2physY.append(ind2physy)
                for j in range(len(x)):
                    xt,yt=outTx.GetInverse().TransformPoint((ind2physX[j],ind2physY[j]))
                    ind2physx,ind2physy=image1.TransformPhysicalPointToIndex((xt,yt))
                    Xt.append(ind2physx)
                    Yt.append(ind2physy)
                y0=[item[0] for item in contours[i]]
                x0=[item[2] for item in contours[i]]
                center_transformed=centroids[i,1]

            file_name=inputDir1+'/Registered_points_'+str(i+1)+'.txt'
            out = open(file_name, 'w')
            for s in range(len(Xt)):
                point=str(Yt[s])+','+str(center_transformed)+','+str(Xt[s])+'\n'  
                out.write(point)
            out.close()
#************************************************************************************************
#**************************** Correspondent points **********************************************
#************************************************************************************************
def corr_points(input_folder,numberof_labels):
    import networkx as nx
    #import multiprocess
    max_dist=1 #2.5 for shoulder
    num_max_len=10
    #pool = multiprocess.Pool(6)
    for L in range(int(numberof_labels)):
        inputDir1 = input_folder+'/label_'+str(L+1) #'input folder'
        os.mkdir(inputDir1+'/contours')
        centroids=np.loadtxt(inputDir1+'/centroid_ponits.txt',delimiter=',')
        contours=[]
        lengths=[]
        for i in range(centroids.shape[0]):
            fname=inputDir1+'/contour_points_'+str(i+1)+'.txt'
            c1=np.loadtxt(fname,delimiter=',')
            c2=c1.tolist()
            contours.append([item[0:3] for item in c2])
    #contours.append([item[0:3] for item in contours1])
            lengths.append(len(contours[i]))
        hight=abs(centroids[0,1]-centroids[-1,1])
        max_len=max(lengths)
        registered_points=[]
        for i in range(centroids.shape[0]-1):
            fname=inputDir1+'/Registered_points_'+str(i+1)+'.txt'
            c1=np.loadtxt(fname,delimiter=',')
            c2=c1.tolist()
            registered_points.append(c2[:])
#----------- Make all contours same size ----------------------------
#----------------- Method 1-----------------------------------------
        contour_interp=[]
        t=int(max_len*1)
        u = np.linspace(0,1+1/t,t)
        steps = 1000 # The more subdivisions the better
        for i in range(centroids.shape[0]):
            x1,y1,z1 = uQuery(np.array(contours[i]),u,steps).T
            #x1,y1,z1 = uQuery3(np.array(contours[i]),t)
            ci=[]
            for j in range(x1.shape[0]):
                ci.append([int(round(x1[j])),int(round(y1[j])),int(round(z1[j]))])
            contour_interp.append(ci)
#----------- Create correspondant points only for good points -----------------
        corr_index=np.zeros([2, max_len, centroids.shape[0]-1])-1
        print(centroids.shape[0]-1)
        for j in range(centroids.shape[0]-1,0,-1):
            if len(contours[j])<len(contours[j-1]):
                #print('j=',j,len(contours[j]))
                for i in range(max_len):
                    ind1,v1=min_dist2(contours[j-1],contour_interp[j-1][i])
                    #print(mehran)
                    corr_index[1,i,centroids.shape[0]-j-1]=i
                    #print('j=',j-1,'ind=',ind1[0])
                    ini_point=contours[j-1][ind1[0]]
                    ind2,v2=min_dist2(registered_points[j-1],ini_point)
                    #print(v2)
                    if v2<=max_dist:
                        ind2,v3=min_dist2(contour_interp[j],contours[j][ind2[0]])
                        corr_index[0,i,centroids.shape[0]-j-1]=ind2[0]
            else:
                #print('j=',j,len(contours[j]))
                for i in range(max_len):
                    ind1,v1=min_dist2(contours[j],contour_interp[j][i])
                    #print(mehran)
                    corr_index[0,i,centroids.shape[0]-j-1]=i
                    #print('j=',j-1,'ind=',ind1[0])
                    ini_point=contours[j][ind1[0]]
                    ind2,v2=min_dist2(registered_points[j-1],ini_point)
                    #print(v2)
                    if v2<=max_dist:
                        ind2,v3=min_dist2(contour_interp[j-1],contours[j-1][ind2[0]])
                        corr_index[1,i,centroids.shape[0]-j-1]=ind2[0]
        #------------ assign interpoints ----------------------------------------
        for j in range(centroids.shape[0]-1):
            fill_vect=np.zeros([max_len])
            if min(corr_index[0,:,j])==-1:
                for k in range(max_len):
                    if corr_index[0,k,j]==-1:
                        best_index=find_non0(corr_index[0,:,j],k)
                        best_index=(best_index>=max_len)*(max_len-1)+(best_index<max_len)*best_index
                        fill_vect[k]=best_index+1
                corr_index[0,:,j]=corr_index[0,:,j]+fill_vect        
            else:
                for k in range(max_len):
                    if corr_index[1,k,j]==-1:
                        best_index=find_non0(corr_index[1,:,j],k)
                        best_index=(best_index>=max_len)*(max_len-1)+(best_index<max_len)*best_index
                        fill_vect[k]=best_index+1
                corr_index[1,:,j]=corr_index[1,:,j]+fill_vect      
#------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-----------Vertical Spline interpolation direction 1 (down to up) with Graph **
        Outputpoint=[]
        Corresponded_Interpolated_points=[]
        G = nx.DiGraph()
        for i in range(max_len): #range(max_len)
            G.clear()
            nod1=corr_index[0,i,0]+(1)*10000
            nod2=corr_index[1,i,0]+(2)*10000
            G.add_nodes_from([nod1,nod2])
            G.add_edge(nod1,nod2)
            nod=[]
            nod.append(corr_index[1,i,0])
            for j in range(1,centroids.shape[0]-1,1): #(centroids.shape[0]-2,-1,-1):# 
                new_nod=[]
                #indx=np.where(corr_index[1,:,j]==corr_index[0,i,j])
                if len(nod)<1:
            #print('stop loop j=',j)
                    break
                for k in range(len(nod)):
                    find_index1=np.where(corr_index[0,:,j]==nod[k])
                    if find_index1[0].shape[0]>0:
                #if indx[0].shape[0]>0:
                #print('j=',j,'find_index1=',find_index1[0].shape[0], 'nod=',nod[k] )
                        for l in range(find_index1[0].shape[0]):
                            indx2=corr_index[1,find_index1[0][l],j]
                            G.add_node(indx2+(j+2)*10000)
                            G.add_edge(nod[k]+(j+1)*10000,indx2+(j+2)*10000 )
                    #print('(nod1, nod2)=',nod[k]+(j+1)*1000,indx2+(j+2)*1000)
                            new_nod.append(indx2)
                    else:
                #print('no point found for nod=',nod[k])
                        break
                nod=[]
                nod=new_nod
            #print(list(G.nodes()))
            leaves = (v for v, d in G.out_degree() if d == 0)
            path_idx=0
            for path in nx.all_simple_paths(G,source=nod1, target=leaves):
                points_for_interpolate=[]
                interpolated_points=[]
                path_idx=path_idx+1
        #print(path)
                for point in path:
                    slice_num=int(point/10000)
                    point_idx=int(point-slice_num*10000)
            #points_for_interpolate.append(contour_interp[slice_num-1][point_idx])
                    points_for_interpolate.append(contour_interp[centroids.shape[0]-slice_num][point_idx])
                if len(points_for_interpolate)==2:
            #---- Method 1 for borders condition -------
                    add1=(np.array(points_for_interpolate[0])+np.array(points_for_interpolate[1]))/2
                    add2=(np.array(points_for_interpolate[0])+add1)/2
                    add3=points_for_interpolate[1]
                    points_for_interpolate[1]=add1.tolist()
                    points_for_interpolate.append(add2.tolist())
                    points_for_interpolate.append(add3)
            #---- Method 2  for borders condition ------
                if len(points_for_interpolate)==3:
                    add1=(np.array(points_for_interpolate[1])+np.array(points_for_interpolate[2]))/2
                    add2=points_for_interpolate[2]
                    points_for_interpolate[2]=add1.tolist()
                    points_for_interpolate.append(add2)
        #print('points for interpolate=',[x,z,y])    
        #cof_mat,u = splprep([x,z,y])
                p_= np.diff(points_for_interpolate,axis=0) # get distances between segments
                m_ = np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
                s_ = np.cumsum(m_)
        #t=3*int(abs(points_for_interpolate[0][1]-points_for_interpolate[-1][0]))
                t=int(round(s_[-1]))
                u = np.linspace(0,1+1/t,t)
                steps = int(num_max_len*hight) # The more subdivisions the better
                x1,y1,z1 = uQuery(np.array(points_for_interpolate),u,steps).T                 # Vertical Interpolation 
                #x1,y1,z1 = uQuery3(np.array(points_for_interpolate),steps) 
                for k in range(x1.shape[0]):
                #interpolated_points.append(np.round(splev(t[j],cof_mat)[:]))
                    interpolated_points.append([x1[k],y1[k],z1[k]])

                list_point=[[int(round(point[0])),int(round(point[1])),int(round(point[2]))] for point in interpolated_points]
 
                Corresponded_Interpolated_points.append(list_point)

        #**** Horizontal spline interpolated contours first direction***
        horizontal_points=[]
        for i in range(int(centroids[0,1]),int(centroids[-1,1]+1),1):
            horizontal_pointi=[]
            for corres_p in Corresponded_Interpolated_points:
                p=np.array(corres_p)
                p1=p[p[:,1]==i,:]
                if p1.shape[0]>0:
                    for j in range(p1.shape[0]):
                        if p1[j,:].tolist() not in horizontal_pointi:
                            horizontal_pointi.append(p1[j,:].tolist())
            if len(horizontal_pointi)>0:
                horizontal_points.append(horizontal_pointi)
        c=0
        for h_points in horizontal_points:
            # add first point to the end of contour to connect all points
            if h_points[-1]!=h_points[0]:
                h_points.append(h_points[0])
            c=c+1
            #print('h_points=',h_points)
            p_= np.diff(h_points,axis=0) # get distances between segments
            m_= np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
            s_= np.cumsum(m_)
            #print(p_,m_,s_)
            #t=3*int(abs(points_for_interpolate[0][1]-points_for_interpolate[-1][0]))
            t=int(round(s_[-1]))+10
            u = np.linspace(0,1+1/t,t)
            steps = int(num_max_len*max_len) # The more subdivisions the better
            x1,y1,z1 = uQuery2(np.array(h_points),u,steps,projection=False).T                   #horizontal interpolation
            #x1,y1,z1 = uQuery3(np.array(h_points),steps)
            H_interpolated_points=[]
            for j in range(x1.shape[0]):
                cr=[x1[j],y1[j],z1[j]]
                if cr not in H_interpolated_points:
                    H_interpolated_points.append(cr)
        # for optimize remove duplicates
            file_name=inputDir1+'/contours/Interpolated1_countor_'+str(c)+'.txt'
            out = open(file_name, 'w')
            for s in H_interpolated_points:
                point=str(int(round(s[0])))+','+str(int(round(s[1])))+','+str(int(round(s[2])))+'\n'  
                out.write(point)
            out.close()
        #--------------------------------------------------------------------------------------------------
        #-------------Vertical Spline interpolation direction 2 (up to down) with Graph--------------------
        #--------------------------------------------------------------------------------------------------
        Outputpoint=[]
        Corresponded_Interpolated_points=[]
        G = nx.DiGraph()
        for i in range(max_len): #range(max_len)
            G.clear()
            nod1=corr_index[1,i,-1]+(1)*10000
            nod2=corr_index[0,i,-1]+(2)*10000
            G.add_nodes_from([nod1,nod2])
            G.add_edge(nod1,nod2)
            nod=[]
            nod.append(corr_index[0,i,-1])
            #print('(nod1, nod2)=',nod1,nod2)
            #indx=np.where(corr_index[0,:,centroids.shape[0]-2]==corr_index[0,i,j])
            for j in range(centroids.shape[0]-3,-1,-1): #(centroids.shape[0]-2,-1,-1):# 
                new_nod=[]
                #indx=np.where(corr_index[1,:,j]==corr_index[0,i,j])
                if len(nod)<1:
                    #print('stop loop j=',j)
                    break
        #print('current nod=',nod, 'lenght(nod)=',len(nod)) 
                for k in range(len(nod)):
                    find_index1=np.where(corr_index[1,:,j]==nod[k])
                    if find_index1[0].shape[0]>0:
                        #if indx[0].shape[0]>0:
                        #print('j=',j,'find_index1=',find_index1[0].shape[0], 'nod=',nod[k] )
                        for l in range(find_index1[0].shape[0]):
                            indx2=corr_index[0,find_index1[0][l],j]
                            G.add_node(indx2+(centroids.shape[0]-j)*10000)
                            G.add_edge(nod[k]+(centroids.shape[0]-j-1)*10000,indx2+(centroids.shape[0]-j)*10000 )
                            #print('(nod1, nod2)=',nod[k]+(j+1)*1000,indx2+(j+2)*1000)
                            new_nod.append(indx2)
                    else:
                        #print('no point found for nod=',nod[k])
                        break
                nod=[]
                nod=new_nod
            #print(list(G.nodes()))
            #print(list(G.edges()))
            leaves = (v for v, d in G.out_degree() if d == 0)
            path_idx=0
            for path in nx.all_simple_paths(G,source=nod1, target=leaves):
                points_for_interpolate=[]
                interpolated_points=[]
                path_idx=path_idx+1
                #print(path)
                for point in path:
                    slice_num=int(point/10000)
                    point_idx=int(point-slice_num*10000)
                    #points_for_interpolate.append(contour_interp[slice_num-1][point_idx])
                    points_for_interpolate.append(contour_interp[slice_num-1][point_idx])
                if len(points_for_interpolate)==2:
                    #---- Method 1 for borders condition -------
                    add1=(np.array(points_for_interpolate[0])+np.array(points_for_interpolate[1]))/2
                    add2=(np.array(points_for_interpolate[0])+add1)/2
                    add3=points_for_interpolate[1]
                    points_for_interpolate[1]=add1.tolist()
                    points_for_interpolate.append(add2.tolist())
                    points_for_interpolate.append(add3)
                    #add=np.array(points_for_interpolate[-1])-0.1
                    #points_for_interpolate.append(add.tolist())
                    #---- Method 2  for borders condition ------
                if len(points_for_interpolate)==3:
                    add1=(np.array(points_for_interpolate[1])+np.array(points_for_interpolate[2]))/2
                    add2=points_for_interpolate[2]
                    points_for_interpolate[2]=add1.tolist()
                    points_for_interpolate.append(add2)
                #print('points for interpolate=',[x,z,y])    
                #cof_mat,u = splprep([x,z,y])
                p_= np.diff(points_for_interpolate,axis=0) # get distances between segments
                m_ = np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
                s_ = np.cumsum(m_)
                #t=3*int(abs(points_for_interpolate[0][1]-points_for_interpolate[-1][0]))
                t=int(round(s_[-1]))
                u = np.linspace(0,1+1/t,t)
                steps = int(num_max_len*hight) # The more subdivisions the better
                x1,y1,z1 = uQuery(np.array(points_for_interpolate),u,steps).T                            # Vertical Interpolation
                #x1,y1,z1 = uQuery3(np.array(points_for_interpolate),steps) 
                for k in range(x1.shape[0]):
                    #interpolated_points.append(np.round(splev(t[j],cof_mat)[:]))
                    interpolated_points.append([x1[k],y1[k],z1[k]])

                list_point=[[int(round(point[0])),int(round(point[1])),int(round(point[2]))] for point in interpolated_points]
        
                # file_name=inputDir1+'/Corresponded_Interpolated_points_'+str(i+1)+'_'+str(path_idx)+'.txt'
                # out = open(file_name, 'w')
                # for s in list_point:
                #     point=str(s[0])+','+str(s[1])+','+str(s[2])+'\n'  
                #     out.write(point)
                # out.close()
                Corresponded_Interpolated_points.append(list_point)
        
        #**** Horizontal spline interpolated contours first direction***
        horizontal_points=[]
        for i in range(int(centroids[0,1]),int(centroids[-1,1]+1),1):
            horizontal_pointi=[]
            for corres_p in Corresponded_Interpolated_points:
                p=np.array(corres_p)
                p1=p[p[:,1]==i,:]
                if p1.shape[0]>0:
                    for j in range(p1.shape[0]):
                        if p1[j,:].tolist() not in horizontal_pointi:
                            horizontal_pointi.append(p1[j,:].tolist())
            if len(horizontal_pointi)>0:
                horizontal_points.append(horizontal_pointi)
        c=0
        for h_points in horizontal_points:
            # add first point to the end of contour to connect all points
            if h_points[-1]!=h_points[0]:
                h_points.append(h_points[0])
            c=c+1
            p_= np.diff(h_points,axis=0) # get distances between segments
            m_ = np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
            s_ = np.cumsum(m_)
    #t=3*int(abs(points_for_interpolate[0][1]-points_for_interpolate[-1][0]))
            t=int(round(s_[-1]))+10
            u = np.linspace(0,1+1/t,t)
            steps = int(num_max_len*max_len) #int(t)  # The more subdivisions the better
            x1,y1,z1 = uQuery2(np.array(h_points),u,steps,projection=False).T                # Horizonal Interpolation
            #x1,y1,z1 = uQuery3(np.array(h_points),steps)
            H_interpolated_points=[]
            for j in range(x1.shape[0]):
                if [int(x1[j]),int(y1[j]),int(z1[j])] not in H_interpolated_points:
                    H_interpolated_points.append([x1[j],y1[j],z1[j]])
            # for optimize remove duplicates
            file_name=inputDir1+'/contours/Interpolated2_countor_'+str(c)+'.txt'
            out = open(file_name, 'w')
            for s in H_interpolated_points:
                #interpolated_points.append(np.round(splev(t[j],cof_mat)[:]))
                #H_interpolated_points.append([x1[j],y1[j],z1[j]])
                point=str(int(round(s[0])))+','+str(int(round(s[1])))+','+str(int(round(s[2])))+'\n'  
                out.write(point)
            out.close()   
#*************************************************************************************************
#******************************** Fill contours **************************************************
#*************************************************************************************************
def fill_contours(label_folder,contours_folder,number_of_labels,output_folder,interp_dirLabel):
    from scipy.ndimage import binary_fill_holes
    for L in range(int(number_of_labels)):
        inputDir1 = label_folder+'/label_'+str(L+1)+'.nii.gz' #'input label'
        label = sitk.ReadImage(inputDir1,sitk.sitkInt16)
        labelArray = sitk.GetArrayFromImage(label)
        inputDir2 = contours_folder+'/label_'+str(L+1)+'/contours' #'input folder'
        #print('inputDir1=',inputDir1,' inputDir2=' ,inputDir2)
        folders = [f for f in os.listdir(inputDir2) if f.endswith('.txt')]
        
        if interp_dirLabel=='Axial':
            i1=1; i2=2; i3=0
        elif interp_dirLabel=='Sagittal':
            i1=2; i2=0; i3=1
        else :#interp_dirLabel=='Coronal'
            i1=2; i2=1; i3=0
        for file_name in folders:
            f_name=inputDir2+'/'+file_name
            with open(f_name, "r") as filestream:
                for line in filestream:
                    currentline = line.split(",")
                    in1=(int(currentline[i1])<0)+(int(currentline[i1])>=0 and int(currentline[i1])<labelArray.shape[0])*int(currentline[i1])+(int(currentline[i1])>=labelArray.shape[0])*(labelArray.shape[0]-1)
                    in2=(int(currentline[i2])<0)+(int(currentline[i2])>=0 and int(currentline[i2])<labelArray.shape[1])*int(currentline[i2])+(int(currentline[i2])>=labelArray.shape[1])*(labelArray.shape[1]-1)
                    in3=(int(currentline[i3])<0)+(int(currentline[i3])>=0 and int(currentline[i3])<labelArray.shape[2])*int(currentline[i3])+(int(currentline[i3])>=labelArray.shape[2])*(labelArray.shape[2]-1)
                    #inter_points.append([int(currentline[0]), int(currentline[1]), int(currentline[2])])        
                    labelArray[in1, in2, in3]=int(L+1)
        #*********************************** Output interpolated labelmap*************************************
        if interp_dirLabel=='Axial':
            for j in range(1,labelArray.shape[0]):
                mat=labelArray[j,:,:]
                labelArray[j,:,:]=filled = binary_fill_holes(mat).astype(int)
        elif interp_dirLabel=='Sagittal':
            for j in range(1,labelArray.shape[2]):
                mat=labelArray[:,:,j]
                labelArray[:,:,j]=filled = binary_fill_holes(mat).astype(int)
        else :#interp_dirLabel=='Coronal'
            for j in range(1,labelArray.shape[1]):
                mat=labelArray[:,j,:]
                labelArray[:,j,:]=filled = binary_fill_holes(mat).astype(int)
        labelArray[labelArray!=0]=int(L+1)
        resultImage = sitk.GetImageFromArray(labelArray)
        resultImage.CopyInformation(label)
        #print(resultImage.GetSpacing())
        l_name=output_folder+'/label_reconstructed_'+str(L+1)+'.nii.gz'
        sitk.WriteImage(resultImage, l_name)
        labelArray=None
#**************************************************************************************************************
#**************************** smoothing the labels ************************************************************
def smooth_label(input_folder,number_of_labels,sigma,output_folder):
    lms = slicer.modules.labelmapsmoothing
    parameters = {}
    parameters['labelToSmooth'] = -1
    parameters['numberOfIterations'] = 10
    parameters['maxRMSError'] = 0.01
    parameters['gaussianSigma'] = sigma
    for L in range(int(number_of_labels)):
        input_volume=slicer.util.loadVolume(input_folder+'/label_reconstructed_'+str(L+1)+'.nii.gz', properties={}, returnNode=False) 
        tmpImageLMS = slicer.vtkMRMLScalarVolumeNode()
        tmpImageLMS.SetName('tmp_LMS')
        slicer.mrmlScene.AddNode(tmpImageLMS)
        output_volume=output_folder+'/label_'+str(L+1)+'.nii.gz'
        parameters['inputVolume']  = input_volume.GetID()
        parameters['outputVolume'] = tmpImageLMS.GetID()
        slicer.cli.run(lms, None, parameters, True)
        #slicer.util.saveNode(tmpImageLMS, output_volume, properties={})
        label = tmpImageLMS.GetImageData()
        dirs = np.zeros([3,3])
        tmpImageLMS.GetIJKToRASDirections(dirs)
        dirs=np.multiply(dirs,np.array([[-1,-1,-1],[-1,-1,-1],[1,1,1]]))
        direction=np.reshape(dirs,(1,9)).tolist()
        origin=tmpImageLMS.GetOrigin()
        labelArray =slicer.util.array(tmpImageLMS.GetID())
        labelArray[labelArray!=0]=int(L+1)
        resultImage = sitk.GetImageFromArray(labelArray)
        resultImage.SetSpacing(tmpImageLMS.GetSpacing())
        resultImage.SetOrigin((origin[0]*-1.0, origin[1]*-1.0,origin[2]) )
        resultImage.SetDirection(direction[0])
        sitk.WriteImage(resultImage, output_volume)
        slicer.mrmlScene.RemoveNode(tmpImageLMS)
        slicer.mrmlScene.RemoveNode(input_volume)
#**************************************************************************************************************
#**************************************** All labels *********************************************
def all_labels(input_folder,number_of_labels,output_folder):
    fname=input_folder+'/label_1.nii.gz'
    label = sitk.ReadImage(fname,sitk.sitkInt16)
    labelArray = sitk.GetArrayFromImage(label)
    label_value=1
    all_label_array=np.zeros((labelArray.shape[0], labelArray.shape[1], labelArray.shape[2]))
    InterSection=np.zeros((labelArray.shape[0], labelArray.shape[1], labelArray.shape[2]))
    for L in range(int(number_of_labels)):
        fname=input_folder+'/label_'+str(L+1)+'.nii.gz'
        label1 = sitk.ReadImage(fname,sitk.sitkInt16)
        label1_array=sitk.GetArrayFromImage(label1)#*int(ll[1])
        intersection=all_label_array*label1_array
        InterSection=InterSection+intersection
        intersection=intersection+1
        intersection[intersection!=1]=0
        all_label_array=label1_array+all_label_array
        all_label_array=all_label_array*intersection
        del label1
        del label1_array
        del intersection    
    resultImage = sitk.GetImageFromArray(all_label_array)
    resultImage.CopyInformation(label)
    output=output_folder+'/all_labels.nii.gz'
    sitk.WriteImage(resultImage, output) 
    
    InterSection[InterSection!=0]=1
    resultImage = sitk.GetImageFromArray(InterSection)
    resultImage.CopyInformation(label)
    l_name=output_folder+'/Intersection.nii.gz'
    sitk.WriteImage(resultImage, l_name)   
#******************************************** My function ****************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#******************** min dist from a line **************************
def dist_p(list1,a,b,c):
    min_dist=10000
    closest_index=0
    print(a,b,c)
    for i in range(len(list1)):
        distp=np.abs(a*list1[i][0]+b*list1[i][2]+c)/np.sqrt(a*a+b*b)
        if distp<min_dist:
            min_dist=distp
            closest_index=i
    x0=(b*(b*list1[closest_index][0]-a*list1[closest_index][2])-a*c)/(a*a+b*b)
    y0=(a*(-b*list1[closest_index][0]+a*list1[closest_index][2])-b*c)/(a*a+b*b)
    #return list1[closest_index][0],list1[closest_index][2]
    return x0, y0
        
#******************** angle distance between two contours************
def angle_dist(list1,list2):
    sum1=0
    for i in range(len(list1)):
        min_ang=360
        for j in range(len(list2)):
            dg=abs(list1[i]-list2[j])
            if dg>180:
                dg=dg-180
            if dg<min_ang:
                min_ang=dg
        sum1=sum1+min_ang
    return sum1        
#******************** find min distance index ***********************
def min_dist(list1,point):
    #print('list=',list1)
    #print('point=',point)
    dists=[]
    for a in list1:
        dists.append(np.sqrt((a[0][0]-point[0])**2 + (a[0][1]-point[1])**2 + (a[0][2]-point[2])**2))
    min_indexes=[i for i, j in enumerate(dists) if j == min(dists)]
    #print('len dist=',len(dists))
    return min_indexes, min(dists)
#********************* find min distance index 2**********************
def min_dist2(list1,point):
    #print('len list=',len(list1))
    #print('point=',point)
    dists=[]
    for a in list1:
        dists.append(np.sqrt((a[0]-point[0])**2 + (a[1]-point[1])**2 + (a[2]-point[2])**2))
    min_indexes=[i for i, j in enumerate(dists) if j == min(dists)]
    #print('len dist=',len(dists))
    #print(dists)
    return min_indexes, min(dists)
#*********************** calculate angle ****************************
def get_angle(c1,c2,b1,b2):
    if(c1 < b1 and c2 > b2): # first quarter 0-90
        angle=np.arctan((b1-c1)/(c2-b2))
    elif(c1 < b1 and c2 < b2): # second quarter 90-180
        angle=np.arctan((b2-c2)/(b1-c1))+np.pi*0.5
    elif(c1 > b1 and c2 < b2): # third quarter 180-270
        angle=np.arctan((c1-b1)/(b2-c2))+np.pi
        #print(borders[k],centroid[j],angle*180/np.pi)
    elif(c1 > b1 and c2 > b2): # forth quarter 270-360
        angle=np.arctan((c2-b2)/(c1-b1))+np.pi*(1.5)
    else:
        angle=(c1 == b1 and c2 > b2)*0+(c1 < b1 and c2 == b2)*np.pi/2+(c1 == b1 and c2 < b2)*np.pi+(c1 > b1 and c2 == b2)*np.pi*1.5
    return angle
#************************* scale respect to center********************
def scale(img,c,gamma,j):
    #from scipy import interpolate 
    #import scipy as sp
    contour1=GetBorders(img,j)
    #print(contour1)
    contour_new=[]
    for p in contour1:
        dx=p[0]-c[0]
        dy=p[2]-c[1]
        contour_new.append([p[0]+int(dx*gamma),j,p[2]+int(dy*gamma)])
    data=np.array(contour_new)
    data=np.transpose(data)
    tck, u = interpolate.splprep([data[0],data[2]], s=0)
    unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)
    #print(np.rint(out).astype(int))
#    data=np.transpose(np.array([np.array(x_new),np.array(y_new)]))
#    data=data[data[:,0].argsort()]  
#    #new_contour.append([x_new,y_new])
#    #new_contour = interp1d(x_new, y_new, kind='cubic')
#    new_length=int(len(contour1)*(gamma+1))
#    new_x_int = np.linspace(data[:,0].min(), data[:,0].max(), new_length)
#    new_y = sp.interpolate.interp1d(np.transpose(data[:,0]), np.transpose(data[:,1]), kind='cubic')(new_x_int)
#    return data
    return contour_new
    
#    return [np.rint(out[0]).astype(int),np.rint(out[1]).astype(int)]

#********************Get borders of slice*****************************
def GetBorders(mat,k):
    list=[]    
    for i in range(mat.shape[0]):
        if (np.sum(mat[i,:])>=1):
            line=mat[i,:].nonzero()
            #print(line)
            #print(i)
            for j in range(len(line[0])):       
                neighbors=GetNeighbor8(mat,[i,line[0][j]])
                if(np.sum(neighbors)<=7):
                    list.append([i,k,line[0][j]])
            #print(list)
    return list
#***********************correspondent list creator***************************
def correspondent_list_creator(list_point,i,subfolder):   
    # for different number of slices must change. here is 11 slices
    c=0;
    #print(list_point[2])
    for ind1 in list_point[0]:
        for ind2 in list_point[1]:
            for ind3 in list_point[2]:
                for ind4 in list_point[3]:
                    for ind5 in list_point[4]:
                        for ind6 in list_point[5]:
                            for ind7 in list_point[6]:
                                for ind8 in list_point[7]:
                                    for ind9 in list_point[8]:
                                        for ind10 in list_point[9]:
                                            for ind11 in list_point[10]:                                                
                                                file_name='correspondent_ponits/'+subfolder+'/correspondent_ponits_'+str(i)+'_'+str(c)+'.txt'
                                                out = open(file_name, 'w')
                                                out.write(str(ind1[0])+','+str(ind1[1])+','+str(ind1[2])+'\n')
                                                out.write(str(ind2[0])+','+str(ind2[1])+','+str(ind2[2])+'\n')
                                                out.write(str(ind3[0])+','+str(ind3[1])+','+str(ind3[2])+'\n')
                                                out.write(str(ind4[0])+','+str(ind4[1])+','+str(ind4[2])+'\n')
                                                out.write(str(ind5[0])+','+str(ind5[1])+','+str(ind5[2])+'\n')
                                                out.write(str(ind6[0])+','+str(ind6[1])+','+str(ind6[2])+'\n')
                                                out.write(str(ind7[0])+','+str(ind7[1])+','+str(ind7[2])+'\n')
                                                out.write(str(ind8[0])+','+str(ind8[1])+','+str(ind8[2])+'\n')
                                                out.write(str(ind9[0])+','+str(ind9[1])+','+str(ind9[2])+'\n')
                                                out.write(str(ind10[0])+','+str(ind10[1])+','+str(ind10[2])+'\n')
                                                out.write(str(ind11[0])+','+str(ind11[1])+','+str(ind11[2])+'\n')
                                            out.close()
                                            c=c+1
    return c                                               
#***************************************** Get Neighbors *****************************************************
def GetNeighbors(img,Ind):
    p,r,c=Ind
    neighborhood=np.zeros([27,1])
    try:
        neighborhood[0] = img[p-1, r-1, c-1]
        neighborhood[1] = img[p-1, r,   c-1]
        neighborhood[2] = img[p-1, r+1, c-1]
    
        neighborhood[ 3] = img[p-1, r-1, c]
        neighborhood[ 4] = img[p-1, r,   c]#dist 1
        neighborhood[ 5] = img[p-1, r+1, c]
    
        neighborhood[ 6] = img[p-1, r-1, c+1]
        neighborhood[ 7] = img[p-1, r,   c+1]
        neighborhood[ 8] = img[p-1, r+1, c+1]

        neighborhood[ 9] = img[p, r-1, c-1]
        neighborhood[10] = img[p, r,   c-1]#dist 1
        neighborhood[11] = img[p, r+1, c-1]

        neighborhood[12] = img[p, r-1, c]#dist 1
        neighborhood[13] = 0 #img[p, r,   c] center
        neighborhood[14] = img[p, r+1, c]#dist 1

        neighborhood[15] = img[p, r-1, c+1]
        neighborhood[16] = img[p, r,   c+1]#dist 1
        neighborhood[17] = img[p, r+1, c+1]

        neighborhood[18] = img[p+1, r-1, c-1]
        neighborhood[19] = img[p+1, r,   c-1]
        neighborhood[20] = img[p+1, r+1, c-1]

        neighborhood[21] = img[p+1, r-1, c]
        neighborhood[22] = img[p+1, r,   c]#dist 1
        neighborhood[23] = img[p+1, r+1, c]

        neighborhood[24] = img[p+1, r-1, c+1]
        neighborhood[25] = img[p+1, r,   c+1]
        neighborhood[26] = img[p+1, r+1, c+1]
    except:
        p,r,c=Ind
    return neighborhood
#------------------------------------------------------
def GetNeighbor8(img,IndMat):
    p,r=IndMat
    neighbor8=np.zeros(9)
    #print([p[0], r[0], c[0]])    
    try:        
        neighbor8[0] = img[p-1, r-1]
        neighbor8[1] = img[p-1, r  ]
        neighbor8[2] = img[p-1, r+1]
    
        neighbor8[3] = img[p, r-1]
        neighbor8[4] = img[p, r  ]   #center
        neighbor8[5] = img[p, r+1]
    
        neighbor8[6] = img[p+1, r-1]
        neighbor8[7] = img[p+1, r  ]
        neighbor8[8] = img[p+1, r+1]
    except:
         p,r=IndMat    
    return neighbor8
#------------------- cubic spline interpolation ----------------
def csvn2(x,y,z,n_points):
    x.append(x[0])
    y.append(y[0])
    
    Px=np.concatenate(([0],x,[0]))
    Py=np.concatenate(([0],y,[0]))
    # interpolation equations
    n=len(x)
    #print('len(x)=',n)
    phi=np.zeros((n+2,n+2))
    for i in range(n):
        phi[i+1,i]=1
        phi[i+1,i+1]=4
        phi[i+1,i+2]=1
# end condition constraints 
    phi[0,0]=-3
    phi[0,2]=3
    phi[0,n-1]=3
    phi[0,n+1]=-3
    phi[n+1,0]=6
    phi[n+1,1]=-12
    phi[n+1,2]=6
    phi[n+1,n-1]=-6
    phi[n+1,n]=12
    phi[n+1,n+1]=-6
# passage matrix
    phi_inv = np.linalg.inv(phi)
# control points
    Qx=6*phi_inv.dot(Px)
    Qy=6*phi_inv.dot(Py)
# all distances between points
    all_dist=[]    
    for i in range(len(x)-1):
        all_dist.append(np.sqrt(abs(x[i]-x[i+1])**2+abs(y[i]-y[i+1])**2))
    dist_vect=np.add.accumulate(all_dist)
    step=sum(all_dist)/(n_points+1)
    step_vect=np.add.accumulate(np.ones(n_points)*step)
    #print('dist_vect=',len(dist_vect))
    #print('step_vect=',len(step_vect))
    t=np.linspace(0,1,num=10)
    xyz=[]
    #for k in range(0,n-1):
    for i in range(n_points):
        try:
            ind=np.where(dist_vect>=step_vect[i])[0][0]
        except:
            ind=np.where(dist_vect>=step_vect[i])[0]
        #print(i,ind, step_vect[i],dist_vect[-1])
        #k=ind-1
        #x_t.append(1.0/6.0*(((1-t)**3)*Qx[k]+(3*t**3-6*t**2+4)*Qx[k+1]+(-3*t**3+3*t**2+3*t+1)*Qx[k+2]+(t**3)*Qx[k+3]))
        #y_t.append(1.0/6.0*(((1-t)**3)*Qy[k]+(3*t**3-6*t**2+4)*Qy[k+1]+(-3*t**3+3*t**2+3*t+1)*Qy[k+2]+(t**3)*Qy[k+3]))
        xyz.append([round(step*x[ind]+(1-step)*x[ind+1]),z,round(step*y[ind]+(1-step)*y[ind+1])])
    #print(type(x_t)    )
    return xyz
#----------------assign interpoints of the target points------------------
def find_non0(V,index):
    iter1=index+1
    lenv=V.shape[0]
    iter1=(iter1>=lenv)*(iter1-lenv)+(iter1<lenv)*iter1
    c1=1
    #print('iter1=',iter1)
    print('index=',index)
    print(V)
    while iter1!=index: # find highest index with not assigned
        if V[iter1]!=-1:
            f_idx=iter1
            f_corr=V[f_idx]
            iter1=index
            break
        iter1=iter1+1
        iter1=(iter1>=lenv)*(iter1-lenv)+(iter1<lenv)*iter1
        c1=c1+1
    iter1=index-1
    iter1=(iter1<0)*(iter1+lenv)+(iter1>=0)*iter1
    c2=1
    print('f_corr=',f_corr)
    print('iter2=',iter1)
    
    while iter1!=index:# find lowest index with not assigned
        if V[iter1]!=-1:
            l_idx=iter1
            l_corr=V[iter1]
            iter1=index
            break
        iter1=iter1-1
        iter1=(iter1<0)*(iter1+lenv)+(iter1>=0)*iter1
        c2=c2+1
    c1=c1-1
    c2=c2-1
    #print(c1,c2)
    dist_corr=abs(f_corr-l_corr)
    dist_idx=abs(c1+c2+1)
    step=dist_corr/dist_idx
    #print('dist_corr=',dist_corr, ' dist_idx=', dist_idx, 'step=',step)
    if f_corr>=l_corr and dist_corr<=lenv/2: #60 76
        best_index=round(f_corr-abs(c1+1)*step)
    elif f_corr<l_corr and dist_corr<=lenv/2:
        best_index=round(f_corr+abs(c1+1)*step)
    else:
        best_index=(c1<=c2)*(f_corr<l_corr)*(f_corr-round((c1+1)*f_corr/dist_idx)) \
                    +(c1<=c2)*(f_corr>l_corr)*(f_corr+c1) \
                    +(c1>c2)*(f_corr<l_corr)*(l_corr+c2) \
                    +(c1>c2)*(f_corr>l_corr)*(l_corr-round((c2+1)*l_corr/dist_idx))
        best_index= (best_index<0)*0+(best_index>=0)*best_index
    print('best_index=',best_index)
    return best_index    
#----------------assign interpoints of the target points------------------
def find_non0_2(V,index):
    lenv=V.shape[0]
    f_idx,c1=find_first_non0(V,index,1)
    f=V[f_idx]
    l_idx,c2=find_first_non0(V,index,-1)
    l=V[l_idx]
    c1=c1-1
    c2=c2-1
    dist_corr=abs(f-l)
    dist_idx=abs(c1+c2+1)
    step=dist_corr/(dist_idx)
    f0_ind, c0=find_first_non0(V,f_idx,1)
    l0_ind, c0=find_first_non0(V,l_idx,-1)
    f0=V[f0_ind]
    l0=V[l0_ind]
    #case1 l0<l<f>f0 -> first-end situation
    if (l0<=l and l<=f and f<=f0) and dist_corr<=(0.5*lenv):
        best_index=(c1<=c2)*round(f-(c1+1)*step)+(c1>c2)*round(l+(c2+1)*step)
    elif (l0>=l and l>=f and f>=f0) and dist_corr<=(0.5*lenv):
        best_index=(c1<=c2)*round(f+(c1+1)*step)+(c1>c2)*round(l-(c2+1)*step)
    elif (l0>=l and l>=f and f<=f0) and dist_corr<=(0.5*lenv):
        best_index=(c1<=c2)*round(f-(c1+1)*step)+(c1>c2)*round(l-(c2+1)*step)
    elif (l0<=l and l<=f and f>=f0) and dist_corr<=(0.5*lenv):
        best_index=(c1<=c2)*round(f+(c1+1)*step)+(c1>c2)*round(l+(c2+1)*step)
    elif (l0<=l and l>=f and f>=f0) and dist_corr<=(0.5*lenv):
        best_index=(c1<=c2)*round(f+(c1+1)*step)+(c1>c2)*round(l+(c2+1)*step)
    elif (l0>=l and l<=f and f<=f0) and dist_corr<=(0.5*lenv):
        best_index=(c1<=c2)*round(f-(c1+1)*step)+(c1>c2)*round(l-(c2+1)*step)
    elif (l0>=l and l<=f and f>=f0) and dist_corr<=(0.5*lenv):
        best_index=(c1<=c2)*round(f+(c1+1)*step)+(c1>c2)*round(l-(c2+1)*step)
    elif (l0<=l and l>=f and f<=f0) and dist_corr<=(0.5*lenv):
        best_index=(c1<=c2)*round(f-(c1+1)*step)+(c1>c2)*round(l+(c2+1)*step)
    else:
        #c1,c2,f,l,l0,f0
        best_index=(c1<=c2)*(f<l)*(f<=f0)*(f-c1) \
                    +(c1<=c2)*(f<l)*(f>f0)*(f+c1) \
                    +(c1<=c2)*(f>l)*(f>=f0)*(f+c1) \
                    +(c1<=c2)*(f>l)*(f<f0)*(f-c1) \
                    +(c1>c2)*(f<l)*(l0<=l)*(l+c2) \
                    +(c1>c2)*(f<l)*(l0>l)*(l-c2) \
                    +(c1>c2)*(f>l)*(l0<l)*(l+c2) \
                    +(c1>c2)*(f>l)*(l0>=l)*(l-c2)
            
    best_index= (best_index<0)*0+(best_index>=0)*best_index
    return best_index
#--------------------------------------------------------------------
def find_first_non0(V,index,direction):
    lenv=V.shape[0]
    if direction==1:
        it=index+1
        it=(it>=lenv)*(it-lenv)+(it<lenv)*it
        c1=1
        while it!=index:
            if V[it]!=-1:
                f_idx=it
                it=index
                break
            it=it+1
            it=(it>=lenv)*(it-lenv)+(it<lenv)*it
            c1=c1+1
    else:
        it=index-1
        it=(it<0)*(it+lenv)+(it>=0)*it
        c1=1
        while it!=index:
            if V[it]!=-1:
                f_idx=it
                it=index
                break
            it=it-1
            it=(it<0)*(it+lenv)+(it>=0)*it
            c1=c1+1
    try:
        dist=c1
        ind=f_idx
    except:
        print('Registration failed. try to add/remove or modify manual contours')
    
    return ind, dist
#---------------- 3D spline interpolation of corresponding points --------------
def uQuery(cv,u,steps=100,projection=True):
    import scipy.interpolate as interpolate
    #import scipy.interpolate as interpolate
    #from scipy import interpolate
    #from scipy.interpolate import splprep, splev
    ''' Brute force point query on spline
        cv     = list of spline control vertices
        u      = list of queries (0-1)
        steps  = number of curve subdivisions (higher value = more precise result)
        projection = method by wich we get the final result
                     - True : project a query onto closest spline segments.
                              this gives good results but requires a high step count
                     - False: modulates the parametric samples and recomputes new curve with splev.
                              this can give better results with fewer samples.
                              definitely works better (and cheaper) when dealing with b-splines (not in this examples)
    '''
    u = np.clip(u,0,1) # Clip u queries between 0 and 1

    # Create spline points
    samples = np.linspace(0,1,steps)
    tck,u_=interpolate.splprep(cv.T,s=0.0)
    p = np.array(interpolate.splev(samples,tck)).T  
    # at first i thought that passing my query list to splev instead
    # of np.linspace would do the trick, but apparently not.    

    # Approximate spline length by adding all the segments
    p_= np.diff(p,axis=0) # get distances between segments
    m = np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
    s = np.cumsum(m) # cumulative summation of magnitudes
    s/=s[-1] # normalize distances using its total length

    # Find closest index boundaries
    s = np.insert(s,0,0) # prepend with 0 for proper index matching
    i0 = (s.searchsorted(u,side='left')-1).clip(min=0) # Find closest lowest boundary position
    i1 = i0+1 # upper boundary will be the next up

    # Return projection on segments for each query
    if projection:
        return ((p[i1]-p[i0])*((u-s[i0])/(s[i1]-s[i0]))[:,None])+p[i0]

    # Else, modulate parametric samples and and pass back to splev
    mod = (((u-s[i0])/(s[i1]-s[i0]))/steps)+samples[i0]
    return np.array(interpolate.splev(mod,tck)).T 
#-----------------------------------------------------------------------------
def uQuery2(cv,u,steps=100,projection=True):
    #from scipy import interpolate 
    import scipy.interpolate as interpolate
    u = np.clip(u,0,1) # Clip u queries between 0 and 1

    # Create spline points
    samples = np.linspace(0,1,steps)
    tck,u_=interpolate.splprep(cv.T,s=0.0,k=2)
    p = np.array(interpolate.splev(samples,tck)).T  
    # at first i thought that passing my query list to splev instead
    # of np.linspace would do the trick, but apparently not.    

    # Approximate spline length by adding all the segments
    p_= np.diff(p,axis=0) # get distances between segments
    m = np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
    s = np.cumsum(m) # cumulative summation of magnitudes
    s/=s[-1] # normalize distances using its total length

    # Find closest index boundaries
    s = np.insert(s,0,0) # prepend with 0 for proper index matching
    i0 = (s.searchsorted(u,side='left')-1).clip(min=0) # Find closest lowest boundary position
    i1 = i0+1 # upper boundary will be the next up
    # Return projection on segments for each query
    if projection:
        return ((p[i1]-p[i0])*((u-s[i0])/(s[i1]-s[i0]))[:,None])+p[i0]

    # Else, modulate parametric samples and and pass back to splev
    mod = (((u-s[i0])/(s[i1]-s[i0]))/steps)+samples[i0]
    return np.array(interpolate.splev(mod,tck)).T 
#-----------------------------------------------------------------------------
def uQuery3(points,num_true_pts):
    #from scipy import interpolate 
    import scipy.interpolate as interpolate
    X=[a[0] for a in points];Y=[a[1] for a in points];Z=[a[2] for a in points]
    tck, u = interpolate.splprep([X,Y,Z], s=2)
    u_fine = np.linspace(0,1,num_true_pts*10)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    return x_fine, y_fine, z_fine 
