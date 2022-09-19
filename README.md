# Spline Semi-automatic Segmentation

<p align="center">
  <img width="128" height="128" src="https://github.com/latimagine/SlicerSpline/blob/main/spline.png?raw=true">
</p>

## Description

A Slicer module for intepolation between manual labels.

## Requirements

The module needs tensorflow, scikit,.... that automatically installs them. If you receive an error, push apply again might solve it.

## Supported System

The module has been tested with Slicer 5.0.3 on Ubuntu 18.04.

## Install

* Download the module
* Add it to 3D Slicer-->Edit-->Application Settings-->Modules
* Add the path of the module to "Additional module paths" and restart (It will appear somewhere in module lists in Example)

## Usage

***NOTE***: If you need a fast processing, you should increase the voxel size using "resample scaler volume" tool.

1- This module needs some manual labels to interpolate them.

2- The labelmap might includes several regions with different labels. They must be from 1 to 100. select the number of labels that you want to interpolate.

3- smoothing factor is to remove isolated voxels might happen during interpolation. We recommend to set it as 1 to 3 if there are a lot of isolated voxels. 

4- The module is able to interpolate in different direction. if you select wrong direction, it may sent error, so you need to select correct direction. 

5- Outputs for this module are the interpolated labelmap and overlap wish shows how much the labels cover each others. Overlap might be very small and may not appear. If overlap is too much you may need to have more manual slices to label before using this module.

6- press apply and wait to finish the process which may takes a few to several minutes depending on resolution of the image.

![alt text](https://github.com/latimagine/SlicerSpline/blob/main/screenshot1.jpg?raw=true)

## Demo

[![Demo CountPages alpha](https://github.com/latimagine/SlicerSpline/blob/main/demo.gif?raw=true)](https://github.com/latimagine/SlicerSpline/blob/main/demo.mp4)

For HR demo, watch the `demo.mp4` video in the root of the project. Sample data (labeled data without volume) is provided in the `data` directory.

## Reference

For more details about the algorithm we refer you to the paper associated to this module:

***Robust Semi-Automatic Segmentation Method: an expert assistant tool for muscles in CT and MR data
Mehran Azimbagirad, Guillaume Dardenne, Douraied Ben salem, Jean-David Werthel
Francois Boux de Cassond, Eric Stindel, Olivier Remy-Neris, and Valerie Burdin***

