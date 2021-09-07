import tensorflow as tf
import numpy as np

import tensorflow.keras.models as KM
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as KB

import re
import os
import datetime
import logging
import sys

import utils


def refine_detections_graph(rois, probs, deltas, window, config):
	"""
	Refine classified proposals and filter overlaps and return final
	detections.
	
	rois: [num_rois, (y1, x1, y2, x2)] in normalized coordinates
	probs: [num_rois, num_classes]. Class probabilities.
	deltas: [num_rois, num_classes, (dy, dx, dh, dw)].
		Class-specific bounding box deltas
	window: (y1, x1, y2, x2) in normalized coordinates.
		The part of the image that contains the image excluding the padding.
		
	Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)]
		where coordinates are normalized.
	"""
	
	# Class IDs per ROI
	# class_ids : [num_rois]  value = 0 ~ num_classes
	class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)

	# Class probability of the top class of each ROI
	# indices.shape = (num_rois, 2)     value = [0~ num_rois-1, index]
	indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
	# class_scores : [num_rois]
	class_scores = tf.gather_nd(probs, indices)
	
	# Class-specific bounding box deltas
	# deltas_specific.shape = (num_rois, (dy, dx, dh, dw))
	deltas_specific = tf.gather_nd(deltas, indices)
	
	# Apply bounding box deltas
	# refined_rois.shape  = (num_rois, (y1, x1, y2, x2)) in normalized coordinates
	refined_rois = utils.apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)

	# Clip boxes to image window
	# refined_rois.shape = (num_rois, (y1, x1, y2, x2))
	refined_rois = utils.clip_boxes_graph(refined_rois, window)
	
	# Filter out background boxes
	# TODO : class_ids == 0 인 것이 background boxes라는 근거가 무엇인가?
	# class_ids == 0 으로 처음에 속성을 설정하는 것인가?
	# probs의 num_rois개의 row에 대응되는 각각의 num_classes개의 Class probabilities중
	# 0번째 class는 background을 의미하기 때문에 class_ids == 0 인 것은 버리는건가?
	keep = tf.where(class_ids > 0)[:, 0]
	if config.DETECTION_MIN_CONFIDENCE:	
		# DETECTION_MIN_CONFIDENCE에 대해 설정값이 있다면 해당 값 이하의 probability를 가진 
		# class_ids는 버린다.
		conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
		# class_scores >= DETECTION_MIN_CONFIDENCE의 교집합
		keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
										tf.expand_dims(conf_keep, 0))
		# 다시 shape 맞춰주기용
		keep = tf.sparse.to_dense(keep)[0]

	# Apply per-class NMS
	# 1. Prepare variables
	pre_nms_class_ids = tf.gather(class_ids, keep)
	pre_nms_scores = tf.gather(class_scores, keep)
	pre_nms_rois = tf.gather(refined_rois,   keep)
	unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]
		
	# tf.map_fn을 사용하면 config전달이 안되기 때문에 여기서 정의
	def nms_keep_map(class_id):
		"""Apply Non-Maximum Suppression on ROIs of the given class."""
		# Indices of ROIs of the given class
		ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
		# Apply NMS
		class_keep = tf.image.non_max_suppression(
					tf.gather(pre_nms_rois, ixs),
					tf.gather(pre_nms_scores, ixs),
					max_output_size	= config.DETECTION_MAX_INSTANCES,
					iou_threshold = config.DETECTION_NMS_THRESHOLD)
		# Map indices
		class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
		# Pad with -1 so returned tensors have the same shape
		gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
		class_keep = tf.pad(class_keep, [(0, gap)],
								mode='CONSTANT', constant_values=-1)
		# Set shape so map_fn() can infer result shape
		class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
		return class_keep

	# 2. Map over class IDs
	nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
						dtype=tf.int64)
	
	# 3. Merge results into one list, and remove -1 padding
	nms_keep = tf.reshape(nms_keep, [-1])
	nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
	
	# 4. Compute intersection between keep and nms_keep
	keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
									tf.expand_dims(nms_keep, 0))
	keep = tf.sparse_tensor_to_dense(keep)[0]

	# Keep top detections
	roi_count = config.DETECTION_MAX_INSTANCES
	class_scores_keep = tf.gather(class_scores, keep)
	num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
	top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
	keep = tf.gather(keep, top_ids)
	
	# Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
	# Coordinates are normalized.
	detections = tf.concat([
		tf.gather(refined_rois, keep),
		tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
		tf.gather(class_scores, keep)[..., tf.newaxis]
		], axis=1)

	# Pad with zeros if detections < DETECTION_MAX_INSTANCES
	gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
	detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
	return detections


class DetectionLayer(KL.Layer):
	"""
	Takes classified proposal boxes and their bounding box deltas 
	and returns the final detection boxes.
	"""
    
	def __init__(self, config, **kwargs):
		super(DetectionLayer, self).__init__(**kwargs)
		self.config = config
		
	def call(self, inputs):
		"""
		inputs : [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
			rpn_rois : [batch, num_rois, (y1, x1, y2, x2)]
			mrcnn_class: [batch, num_rois, NUM_CLASSES] classifier probabilities
			mrcnn_bbox: [batch, num_rois, NUM_CLASSES, (dy, dx, dh, dw)]
				Deltas to apply to proposal boxes
			input_image_meta
		"""
		rois = inputs[0]
		mrcnn_class = inputs[1]
		mrcnn_bbox = inputs[2]
		image_meta = inputs[3]
        
		# Get windows of images in normalized coordinates.
		# Windows are the area in the image that excludes the padding.
		# Use the shape of the first image in the batch to normalize the window
		# because we know that all images get resized to the same size.
		image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
		# window.shape = (batch, (y1, x1, y2, x2))  in normalized coordinates
		window = utils.norm_boxes_graph(parse_image_meta_graph(image_meta)['window'],
										image_shape[:2])
		
		# Run detection refinement graph on each item in the batch
		# detections_batch.shape = (num_detections, (y1, x1, y2, x2, class_id, score))
		detections_batch = utils.batch_slice(
			[rois, mrcnn_class, mrcnn_bbox, window],
			lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
			self.config.IMAGES_PER_GPU)
		
		# Reshape output
		# [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] 
		# in normalized coordinates
		return tf.reshape(
			detections_batch,
			[self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])



def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
	"""
	Mask binary cross-entropy loss for the masks head.
	
	target_mask:	[batch, num_rois, height, width]
	target_class_ids:	[batch, num_rois], Integer class IDs.
	pred_masks: [batch, num_rois, height, width, num_classes]
	"""
	
	# Reshape for simplicity. Merge first two dimensions into one.
	# target_class_ids : [N] 	N == batch * num_rois
	target_class_ids = tf.reshape(target_class_ids, (-1,))
	mask_shape = tf.shape(target_masks)
		
	# target_masks.shape = (N, height, width)
	target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))

	pred_shape = tf.shape(pred_masks)
	# pred_masks.shape = (N, height, width, num_classes)
	pred_masks = tf.reshape(pred_masks,(-1, pred_shape[2], pred_shape[3], pred_shape[4]))

	# Permute predicted masks to [N, num_classes, height, width]
	pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])
		
	# Only positive ROIs contribute to the loss.
	# positive_ix : [positive_N].  positive_N == batch * num_positive_rois
	positive_ix = tf.where(target_class_ids > 0)[:, 0]
	# positive_class_ids.shape (positive_N)
	positive_class_ids = tf.cast(
		tf.gather(target_class_ids, positive_ix), tf.int64)
	# indices.shape = (positive_N, 2)
	indices = tf.stack([positive_ix, positive_class_ids], axis=1)
		
	# Gather the masks (predicted and true) that contribute to loss
	# y_true.shape = (positive_N, height, width)
	y_true = tf.gather(target_masks, positive_ix)
	# y_pred.shape = (positive_N, height, width)
	y_pred = tf.gather_nd(pred_masks, indices)
		
	# Compute binary cross entropy.
	# If no positive ROIs, then return 0.
	loss = KB.switch(tf.size(y_true) > 0,
					KB.binary_crossentropy(target=y_true, output=y_pred),
					tf.constant(0.0))
		
	loss = tf.reduce_mean(loss)
	return loss

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
	"""
	Loss for Mask R-CNN bounding box refinement.
	
	target_bbox:	[batch, num_rois, (dy, dx, dh, dw]
	target_class_ids:	[batch, num_rois], Integer class IDs.
	pred_bbox: [batch, num_rois, NUM_CLASSES, (dy, dx, dh, dw)]
	"""
		
	# Reshape to merge batch and roi dimensions for simplicity.
	# target_class_ids : [N]  N == batch * num_rois
	target_class_ids = tf.reshape(target_class_ids, (-1,))
	# target_bbox.shape = (N, (dy, dx, dh, dw)
	target_bbox = tf.reshape(target_bbox, (-1, 4))
	# pred_bbox.shape = (N, NUM_CLASSES, (dy, dx, dh, dw))
	pred_bbox = tf.reshape(pred_bbox, (-1, pred_bbox.shape[2], 4))

	# Only positive ROIs contribute to the loss.
	# positive_roi_ix.shape = (positive_N).  positive_N == batch * num_positive_rois
	positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
	# positive_roi_class_ids.shape  = (positive_N)
	positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
	
	# indices.shape = (positive_N, 2)
	indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
	
	# Gather the deltas (predicted and true) that contribute to loss
	# target_bbox.shape = (positive_N, (dy, dx, dh, dw))
	target_bbox = tf.gather(target_bbox, positive_roi_ix)
	# pred_bbox.shape = (positive_N, (dy, dx, dh, dw))
	pred_bbox = tf.gather_nd(pred_bbox, indices)
	
	# Smooth-L1 Loss
	loss = KB.switch(tf.size(target_bbox) > 0,
					smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
					tf.constant(0.0))
	

	loss = KB.mean(loss)
	return loss

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
						active_class_ids):
	"""
	Loss for the classifier head of Mask RCNN.
	target_class_ids :	[batch, num_rois], Integer class IDs.
	pred_class_logits : [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
	active_class_ids : [batch, NUM_CLASSES]
		value = 1
		
	"""
	
	target_class_ids = tf.cast(target_class_ids, 'int64')
	
	# Find predictions of classes that are not in the dataset.
	# pred_class_ids : [batch, num_rois]
	pred_class_ids = tf.argmax(pred_class_logits, axis=2)
	# TODO: Update this line to work with batch > 1.
	# right now it assumes all images in a batch have the same active_class_ids
	# pred_active : [batch, num_rois]
	pred_active = tf.gather(active_class_ids[0], pred_class_ids)
	
	# Loss
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=target_class_ids, logits=pred_class_logits)
	
	# Erase losses of predictions of classes that are not in the active classes of the image.
	loss = loss * pred_active
	
	# Compute loss mean. 
	# Computer loss mean. Use only predictions that contribute
	# to the loss to get a correct mean.
	loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
	
	return loss


def smooth_l1_loss(y_true, y_pred):
	"""
	Implements Smooth-L1 loss.
	y_true and y_pred are typically: [N, 4], but could be any shape.
	
	e.g. 
	y_true = target_bbox : [target_anchors, (dy, dx, dh, dw)]
	y_pred = rpn_bbox : [target_anchors, (dy, dx, dh, dw)] 
	
	Return:
		loss : [N, 4]
	"""
	
	
	diff = tf.abs(y_true - y_pred)
	less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
	loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
	return loss

def batch_pack_graph(x, counts, num_rows):
	"""
	Picks different number of values from each row in x depending on the values in counts.
	x : [batch, num_boxes, 4]
		4 : (dy, dx, dh, dw)
	counts : [batch], integer 
	num_rows : IMAGES_PER_GPU
	
	Return : [count_1 + count_2 + ... + count_num_rows, 4]
		4 : (dy, dx, dh, dw)
	"""
		
	outputs = list()
	
	for i in range(num_rows):
		outputs.append(x[i, :counts[i]])
		
	
	return tf.concat(outputs, axis=0)

def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
	"""
	Return the RPN bounding box loss graph.
	
	target_bbox: [batch, max positive anchors, (dy, dx, dh, dw)].
			Uses 0 padding to fill in unsed bbox deltas.
	rpn_match: [batch, anchors, 1]. 
			Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
	rpn_bbox : [batch, anchors, (dy, dx, dh, dw)] 
	"""
	
	# Positive anchors contribute to the loss, 
	# but negative and neutral anchors (match value of 0 or -1) don't.
	# rpn_match : [batch, anchors]
	rpn_match = tf.squeeze(rpn_match, -1)
	indices = tf.where(tf.equal(rpn_match, 1))
	
	# Pick bbox deltas that contribute to the loss
	# rpn_bbox : [batch * positive_anchors, (dy, dx, dh, dw)] 
	# == [target_anchors, (dy, dx, dh, dw)]
	rpn_bbox = tf.gather_nd(rpn_bbox, indices)
	
	# Trim target bounding box deltas to the same length as rpn_bbox.
	# batch_counts : [batch]. 각 batch마다 존재하는 anchors들 중에서 1인 것의 개수
	batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
	# target_bbox : [count_1 + count_2 + ... + count_IMAGES_PER_GPU, (dy, dx, dh, dw)]
	#  == [target_anchors, (dy, dx, dh, dw)]
	target_bbox = batch_pack_graph(target_bbox, batch_counts,
								   config.IMAGES_PER_GPU)
	
	# loss : [target_anchors, 4]
	loss = smooth_l1_loss(target_bbox, rpn_bbox)
	
	# loss : float value
	loss = KB.switch(tf.size(loss) > 0, KB.mean(loss), tf.constant(0.0))

	return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
	"""
	RPN anchor classifier loss.
	
	rpn_match: [batch, anchors, 1]. Anchor match type. 
	1=positive,  -1=negative, 0=neutral anchor.
	rpn_class_logits: [batch, anchors, (background_logit, foreground_logit)]
		RPN classifier logits for BG/FG.
	"""
	
	# Squeeze last dim to simplify
	# [batch, anchors]
	rpn_match = tf.squeeze(rpn_match, -1)
	# Get anchor classes. Convert the -1/+1 match to 0/1 values.
	# value = 1 인 것만 추려냄
	anchor_class = tf.cast(KB.equal(rpn_match, 1), tf.int32)
	
	# Positive and Negative anchors contribute to the loss, but neutral anchors don't
	# neutral anchors : match value = 0
	# value = 1 or -1 인 것들의 indices
	# indices.shape = (batch - N, anchors - N)
	indices = tf.where(KB.not_equal(rpn_match, 0))
	
	# Pick rows that contribute to the loss and filter out the rest.
	# rpn_class_logits = (batch, anchors, [background_logit, foreground_logit])
	rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
	
	# Cross entropy loss
	# loss = (batch, anchors)
	loss = KB.sparse_categorical_crossentropy(target=anchor_class,
											output=rpn_class_logits,
											from_logits=True)
	
	# loss : float value
	# TODO : tf.reduce_mean과 뭐가 다른지 확인해보자.
	loss = KB.switch(tf.size(loss) > 0, KB.mean(loss), tf.constant(0.0))
	
	return loss



def build_fpn_mask_graph(rois, feature_maps, image_meta, num_classes,
                         pool_size = 14, train_bn=True):
	"""
	Builds the computation graph of the mask head of Feature Pyramid Network.
	
	rois: [batch, num_rois, (y1, x1, y2, x2)]
		Proposal boxes in normalized coordinates.
	feature_maps : [P2, P3, P4, P5]		FPN으로부터 계산한 featrue map
	image_meta : [batch, IMAGE_META_SIZE]
	pool_size: The width of the square feature map generated from ROI Pooling.
		pool_size = MASK_POOL_SIZE
	train_bn: Boolean. Train or freeze Batch Norm layers
	
	Returns: Masks [batch, num_rois, (MASK_POOL_SIZE - 1)*2 + 2, (MASK_POOL_SIZE - 1)*2 + 2, num_classes]
	"""

	## ROI Pooling
	# x.shape = (batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels)
	x = KL.PyramidROIAlign([pool_size, pool_size],
						name="roi_align_mask")([rois, image_meta] + feature_maps)

	## Conv layers
	x = KL.TimeDistributed(KL.Conv2D(256, 3, padding="same"), name="mrcnn_mask_conv1")(x)
	x = KL.TimeDistributed(KL.BatchNormalization(), 
						name='mrcnn_mask_bn1')(x, training=train_bn)
	# x : [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, 256]
	x = KL.Activation('relu')(x)
	
	x = KL.TimeDistributed(KL.Conv2D(256, 3, padding="same"),
						name="mrcnn_mask_conv2")(x)
	x = KL.TimeDistributed(KL.BatchNormalization(),
						name='mrcnn_mask_bn2')(x, training=train_bn)
	# x : [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, 256]
	x = KL.Activation('relu')(x)
	
	x = KL.TimeDistributed(KL.Conv2D(256, 3, padding="same"),
						name="mrcnn_mask_conv3")(x)
	x = KL.TimeDistributed(KL.BatchNormalization(),
						name='mrcnn_mask_bn3')(x, training=train_bn)
	# x : [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, 256]
	x = KL.Activation('relu')(x)

	x = KL.TimeDistributed(KL.Conv2D(256, 3, padding="same",
						name="mrcnn_mask_conv4"))(x)
	x = KL.TimeDistributed(KL.BatchNormalization(),
						name='mrcnn_mask_bn4')(x, training=train_bn)
	# x : [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, 256]
	x = KL.Activation('relu')(x)

	# x = [batch, num_rois, (MASK_POOL_SIZE - 1)*2 + 2, (MASK_POOL_SIZE - 1)*2 + 2, 256]
	x = KL.TimeDistributed(KL.Conv2DTranspose(256, 2, strides=2, activation="relu"),
						name="mrcnn_mask_deconv")(x)
		
	# x = [batch, num_rois, (MASK_POOL_SIZE - 1)*2 + 2, (MASK_POOL_SIZE - 1)*2 + 2, num_classes]
	x = KL.TimeDistributed(KL.Conv2D(num_classes, 1, activation="sigmoid"),
							name="mrcnn_mask")(x)
	
	
	return x


def log2_graph(x):
	"""Implementation of Log2. TF doesn't have a native implementation."""
	return tf.math.log(x) / tf.math.log(2.0)

class PyramidROIAlign(KL.Layer):
	"""
	Implements ROI Pooling on multiple levels of the feature pyramid.
	"""
		
	def __init__(self, pool_shape, **kwargs):
		"""
		pool_size : The width of the square feature map generated from ROI Pooling.
		"""
		super(PyramidROIAlign, self).__init__(**kwargs)
		self.pool_shape = tuple(pool_shape)
		
	def call(self, inputs):
		"""
		inputs:
			[rois, image_meta, P2, P3, P4, P5]
			rois : [batch, rois_count, (y1, x1, y2, x2)]  관심영역
			image_meta : [batch, IMAGE_META_SIZE]      		
			P2, P3, P4, P5 : Each is [batch, height, width, channels] 
				FPN으로부터 계산한 featrue map
				
		Output:
		Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
		The width and height are those specific in the pool_shape in the layer
		constructor.
		"""

		# Crop boxes.shape = (batch, num_boxes, (y1, x1, y2, x2)) in normalized coords
		# num_boxes == rois_count
		boxes = inputs[0]
		
		# Image meta
		# Holds details about the image. See compose_image_meta()
		image_meta = inputs[1]
		
		# Feature Maps. 
		# List of feature maps from different level of the feature pyramid.
		feature_maps = inputs[2:]

		## Assign each ROI to a level in the pyramid based on the ROI area.
		# y1.shape = (batch, num_boxes)  same other coordinates
		y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
		h = y2 - y1
		w = x2 - x1
		# Use shape of first image.
		# Images in a batch must have the same size.
		# image_shape.shape = (H, W, C) after resizing and padding
		image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
		
		# Equation 1 in the Feature Pyramid Networks paper. 
		# Account for the fact that our coordinates are normalized here.
		# e.g. a 224x224 ROI (in pixels) maps to P4
		image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
		# roi_level.shape = (batch, num_boxes, 1), value = roi_level
		roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
		# roi_level의 범위  		2 < roi_level < 5 
		roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
		# roi_level.shape (batch, num_boxes),	 value = roi_level
		# 각각의 roi box에 대해서 알맞은 lavel compute
		roi_level = tf.squeeze(roi_level, 2)

		## Loop through levels and apply ROI pooling to each. P2 to P5.
		# [Result_1, Result_2, Result_3, Result_4]
		# 각 Result_1는 4-D
		# len(pooled) = 4가 될거임 (2~6)
		pooled = []
		
		# [[indices of batch, indices of num_boxes], 
		#  [indices of batch, indices of num_boxes], ...]  
		# len(box_to_level) = 4가 될거임 (2~6)
		box_to_level = []	
		
		for i, level in enumerate(range(2, 6)):
			# ix : [indices  batch, indices of num_boxes]
			ix = tf.where(tf.equal(roi_level, level))
			# level_boxes.shape = (indices of batch, indices of num_boxes)
			# 알맞은 lavel(2~5)에 대응되는 roi boxes
			level_boxes = tf.gather_nd(boxes, ix)
			
			# Box indices of num_boxes for crop_and_resize.
			box_indices = tf.cast(ix[:, 0], tf.int32)
			
			# Keep track of which box is mapped to which level
			box_to_level.append(ix)
			
			# Stop gradient propogation to ROI proposals
			level_boxes = tf.stop_gradient(level_boxes)
			box_indices = tf.stop_gradient(box_indices)
			
			# Crop and Resize
			# 4개의 위치를 샘플링하는 Bilinear Interpolation을 사용하는 것이 가장 효율적
			# tf.image.crop_and_resize를 통해 구현
			# Result.shape = (batch * num_boxes, pool_height, pool_width, channels)
			pooled.append(tf.image.crop_and_resize(
				feature_maps[i], level_boxes, box_indices, self.pool_shape,
				method="bilinear"))

		## Pack pooled features into one tensor
		# pooled.shape = (batch * num_boxes * 4, pool_height, pool_width, channels)
		pooled = tf.concat(pooled, axis=0)
		
		## Pack box_to_level mapping into one array and add another
		# box_to_level.shape = (batch indices * 4, num_boxes indices)
		box_to_level = tf.concat(box_to_level, axis=0)
		
		## pooled boxes의 순서를 나타내는 column 생성
		# pooled boxes의 순서를 나타내는 column을 만들기 위해 expand_dims
		# box_range.shape = (batch indices * 4 , 1)
		box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
		# box_to_level[:, -1] 은 해당 boxes의 index를 띄고 있다
		# range = 0부터 (batch * 4) -1 까지
		# box_to_level.shape = (batch indices * 4, num_boxes indices + 1)
		box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
								axis=1)

		## pooling된 features를 boxes 순서와 일치하도록 재정렬
		## box_to_level를 batch순으로 sort후 box index순으로 sort
		# batch순 : 비교하기 쉽게 값을 크게 키운 후
		# sorting_tensor.shape = (batch * 4)
		sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
		# ix.shape = (batch * 4)  value = 상위 값을 가진 batch * 4개 boxes의 indices
		ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
		# ix.shape = (batch * 4)
		ix = tf.gather(box_to_level[:, 2], ix)
		# pooled.shape = (batch * 4, pool_height, pool_width, channels)
		# batch * num_boxes * 4 개 중에서 상위 batch * 4개를 추려냄
		pooled = tf.gather(pooled, ix)
			
		# Re-add the batch dimension
		# shape.shape = (batch, num_boxes, pool_height, pool_width, channels)
		shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
		
		# ???
		# TODO : 위의 batch * 4가 아니라 batch * num_boxes 여야 한다.
		# 그러기 위해서 roi_level 부터 [batch, num_boxes]이 아니라
		# [batch * num_boxes] 이여야 한다.
		# pooled : [batch, num_boxes, pool_height, pool_width, channels]
		pooled = tf.reshape(pooled, shape)
		
		return pooled


def fpn_classifier_graph(config, rois, feature_maps, image_meta, pool_size, 
                         num_classes, train_bn=True,
                         fc_layers_size=1024):
	"""
	FPN의 classifier, regressor heads의 계산 graph를 Build한다.
	
	rois: [batch, rois_count, (y1, x1, y2, x2)]  관심영역
	feature_maps : [P2, P3, P4, P5]		FPN으로부터 계산한 featrue map
	image_meta : [batch, IMAGE_META_SIZE]
	pool_size: The width of the square feature map generated from ROI Pooling.
	train_bn: Boolean. Train or freeze Batch Norm layers
	fc_layers_size: Size of the 2 FC layers
	
	Returns:
		logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
		probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
		bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, dh, dw)] 
			Deltas to apply to proposal boxes
	"""

	## ROI Pooling
	# x.shape = (batch, num_rois, POOL_SIZE, POOL_SIZE, channels)
	x = PyramidROIAlign([pool_size, pool_size],
						name="roi_align_classifier")([rois, image_meta] + feature_maps)

	## Two 1024 FC layers (implemented with Conv2D for consistency)
	x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, pool_size), name="mrcnn_class_conv1")(x)
	x = KL.TimeDistributed(KL.BatchNormalization(),
						name='mrcnn_class_bn1')(x, training=train_bn)
	# x : [batch, num_rois, 1, 1, fc_layers_size]
	x = KL.Activation('relu')(x)
	
	x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, 1), name="mrcnn_class_conv2")(x)
	x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
	# x : [batch, num_rois, 1, 1, fc_layers_size]
	x = KL.Activation('relu')(x)
	
	# shared : [batch, num_rois, fc_layers_size]
	shared = KL.Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2),
					name="pool_squeeze")(x)

	## Classifier head
	# logits
	mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
											name='mrcnn_class_logits')(shared)
	
	# probs
	# mrcnn_probs : [batch, num_rois, num_classes]
	mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
									name="mrcnn_class")(mrcnn_class_logits)
	
	
	## Regression head
	# x : [batch, num_rois, NUM_CLASSES * (dy, dx, dh, dw)]
	x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
						name='mrcnn_bbox_fc')(shared)
	# Reshape 
	s = KB.int_shape(x)
	# mrcnn_bbox : [batch, num_rois, NUM_CLASSES, (dy, dx, dh, dw)]
	mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
		
	return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def overlaps_graph(boxes1, boxes2):
	"""
	Computes IoU overlaps between two sets of boxes.
	boxes1 : [N1, (y1, x1, y2, x2)]
	boxes2 : [N2, (y1, x1, y2, x2)]
		
	Return:
		overlaps : [N1, N2]
	"""
		
	## 1. Tile boxes2 and repeat boxes1. 
	## 반복문 없이 boxes1과 boxes2를 비교하기 위해
	# b1 : [N1 * N2, 1, (y1, x1, y2, x2)]
	# e.g.	[[y1_1, x1_1, y2_1, x2_1]				==  1 iter start
	# 		 [y1_2, x1_2, y2_2, x2_2]
	#		 ...
	#		 [y1_N2, x1_N2, y2_N2, x2_N2]            == 1 iter end
	# 		 [y1_1, x1_1, y2_1, x2_1]				 == 2 iter start
	# 		 ...
	# 		 [y1_N2, x1_N2, y2_N2, x2_N2]]			== N1 iter end
	# Why didn't  b1 = tf.tile(boxes1, [tf.shape(boxes2)[0], 1]) ??
	b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), 
					[1, 1, tf.shape(boxes2)[0]]), [-1, 4])
	# b2 : [N2 * N1, (y1, x1, y2, x2)] 
	# e.g.	[[y1_1, x1_1, y2_1, x2_1]				==  1 iter start
	# 		 [y1_2, x1_2, y2_2, x2_2]
	#		 ...
	#		 [y1_N2, x1_N2, y2_N2, x2_N2]            == 1 iter end
	# 		 [y1_1, x1_1, y2_1, x2_1]				 == 2 iter start
	# 		 ...
	# 		 [y1_N2, x1_N2, y2_N2, x2_N2]]			== N1 iter end
	b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
	
	## 2. Compute intersections
	# b1_y1 : [N1 * N2, y1], b1_x1: [N1 * N2, x1] ...
	b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
	# b2_y1 : [N2 * N1, y1], b2_x1: [N2 * N1, x1] ...
	b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
	y1 = tf.maximum(b1_y1, b2_y1)
	x1 = tf.maximum(b1_x1, b2_x1)
	y2 = tf.minimum(b1_y2, b2_y2)
	x2 = tf.minimum(b1_x2, b2_x2)
	# intersection : [N1 * N2, intersection]
	intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
	
	## 3. Compute unions
	b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
	b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
	# union : [N1 * N2, union]
	union = b1_area + b2_area - intersection
	
	## 4. Compute IoU and reshape to [boxes1, boxes2]
	iou = intersection / union
	# overlaps [N1, N2]
	overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
	
	return overlaps

def trim_zeros_graph(boxes, name='trim_zeros'):
	"""
	boxes는 [N, 4] shape이고, 몇몇 N은 [0, 0, 0, 0]인 경우가 있다. 
	이런 경우의 N은 아예 제거한다.
	
	boxes: [N, 4] matrix of boxes.
	
	Return:
		boxes: [N, 4] matrix of boxes.
		non_zeros: [N] boxes중에서 사용가능한 rows만 True로 표시한 boolean mask
	"""
	non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
	boxes = tf.boolean_mask(boxes, non_zeros, name=name)
	
	return boxes, non_zeros

def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
	"""
	Generates detection targets for one image.
	Subsamples proposals and generates target class IDs, bounding box deltas, and masks for each.
	
	proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)]
			in normalized coordinates.
	gt_class_ids : [MAX_GT_INSTANCES] int class IDs
	gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] 
			in normalized coordinates.
	gt_masks: [height, width, MAX_GT_INSTANCES] 
			of boolean type.
	
	Returns:
		rois: [rois_count, (y1, x1, y2, x2)] in normalized coordinates
		class_ids: [rois_count]. Integer class IDs. Zero padded.
		deltas: [rois_count, (dy, dx, dh, dw)]
		masks: [rois_count, height, width].
			Masks cropped to bbox boundaries and resized to neural network output size.
			TRAIN_ROIS_PER_IMAGE =< rois_count
	
	Note: Returned arrays might be zero padded if not enough target ROIs.
	"""
	## Assertions 해당 없음
	# POST_NMS_ROIS_TRAINING > 0 이 아니면 proposals의 tensor항목을 출력
	asserts = [ tf.Assert(tf.greater(tf.shape(proposals)[0], 0),
				[proposals], name="roi_assertion"), 
			]
	# POST_NMS_ROIS_TRAINING == 0 인 경우에
	# proposals 각 항목을 다시 proposals로 재 할당 
	with tf.control_dependencies(asserts):
		proposals = tf.identity(proposals)

	## Remove zero padding
	# proposals.shape  = (POST_NMS_ROIS_TRAINING - N_p, (y1, x1, y2, x2))
	# N_p is number of removed indices from proposals
	proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
	# gt_boxes.shape = (MAX_GT_INSTANCES - N_g, (y1, x1, y2, x2)) 
	#	 N_g is number of removed indices from gt_boxes
	# non_zeros.shape = (MAX_GT_INSTANCES) 
	# 	gt_boxes중 사용가능한 rows만 True로 표시한 boolean mask
	gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
	# gt_masks.shape = (height, width, MAX_GT_INSTANCES - N_g)
	gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
						name="trim_gt_masks")
	
	# POST_NMS_ROIS_TRAINING - N_p 는 POST_NMS_ROIS_TRAINING으로 표현하겠음
	# MAX_GT_INSTANCES - N_g 는 MAX_GT_INSTANCES으로 표현하겠음

	## Compute overlaps matrix 
	# overlaps.shape = (POST_NMS_ROIS_TRAINING, MAX_GT_INSTANCES)
	# value : iou between proposals and gt_boxes
	overlaps = overlaps_graph(proposals, gt_boxes)
		
	## Determine positive and negative ROIs [foreground ROIs, background ROIs]
	# roi_iou_max.shape = (POST_NMS_ROIS_TRAINING)
	roi_iou_max = tf.reduce_max(overlaps, axis=1)
	# 1. Positive ROIs are those with >= 0.5 IoU with a GT box
	positive_roi_bool = (roi_iou_max >= 0.5)
	positive_indices = tf.where(positive_roi_bool)[:, 0]
	# 2. Negative ROIs are those with < 0.5 with every GT box. 
	negative_roi_bool = (roi_iou_max < 0.5)
	negative_indices = tf.where(negative_roi_bool)[:, 0]

	## Subsample ROIs. Aim for 33% positive
	# Positive ROIs
	positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
	# positive_roi중에서 ROI_POSITIVE_RATIO비율만큼 random하게 뽑는다.
	positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
	positive_count = tf.shape(positive_indices)[0]
	
	# Negative ROIs. Add enough to maintain positive:negative ratio.
	r = 1.0 / config.ROI_POSITIVE_RATIO
	negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
	# positive_roi중에서 1-ROI_POSITIVE_RATIO비율만큼 random하게 뽑는다.
	negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
	
	# Gather selected ROIs
	# positive_rois.shape = (positive_count, (y1, x1, y2, x2))
	# negative_rois.shape = (negative_count, (y1, x1, y2, x2))
	positive_rois = tf.gather(proposals, positive_indices)
	negative_rois = tf.gather(proposals, negative_indices)

	## Assign positive ROIs to GT boxes.
	# positive_overlaps.shape = (POST_NMS_ROIS_TRAINING, positive_count)
	positive_overlaps = tf.gather(overlaps, positive_indices)
	# false_fn : positive_overlaps 전부 그대로 roi_gt_box_assignment에 assign
	# true_fn : roi_gt_box_assignment : shape = [positive_count] , 	value = indices
	roi_gt_box_assignment = tf.cond(
		tf.greater(tf.shape(positive_overlaps)[1], 0),
		true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
		false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
	)
		
	# gt_boxes.shape = (MAX_GT_INSTANCES, (y1, x1, y2, x2))
	# roi_gt_boxes = (positive_count, (y1, x1, y2, x2))
	# iou가 가장 높은 coordinate만 추려낸다.
	roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
	# roi_gt_class_ids.shape = (positive_count)
	# iou가 가장 높은 Bbox에 해당하는 class_ids만 추려낸다.
	roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

	## Compute bbox refinement for positive ROIs
	# deltas.shape = (positive_count, (dy, dx, dh, dw))
	deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
	deltas /= config.BBOX_STD_DEV

	## Assign positive ROIs to GT masks
	# gt_masks.shape (height, width, MAX_GT_INSTANCES)
	# transposed_masks.shape = (MAX_GT_INSTANCES - N, height, width)
	transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
	# Pick the right mask for each ROI
	# roi_masks.shape =  (positive_count, height, width) 
	roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
	######### 여기 roi_masks 에러나는지 확인. ##########
	# 에러난다면 roi_iou_max = tf.reduce_max(overlaps, axis=1) 에서 axis를 0으로 바꿔야한다.
	# MAX_GT_INSTANCES => positive_count 이여야 한다.
	# transposed_masks => roi_gt_box_assignment
	
	## Compute mask targets
	# boxes.shape = (positive_count, (y1, x1, y2, x2))
	boxes = positive_rois
	# box_ids.shape = (0, 1, ..., positive_count -1)
	box_ids = tf.range(0, tf.shape(roi_masks)[0])
	# masks.shape (positive_count, MASK_SHAPE[0], MASK_SHAPE[1], 1)
	masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), 
									boxes, box_ids,
									config.MASK_SHAPE)
	# masks.shape = (positive_count, MASK_SHAPE[0], MASK_SHAPE[1])
	masks = tf.squeeze(masks, axis=3) 

	## Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
	# pixels값 0.5를 기준으로 낮으면 0, 높으면 1의 값을 assign
	masks = tf.round(masks)
	
	## Append negative ROIs and pad bbox deltas and masks that
	# rois.shape = (positive_count + negative_count, (y1, x1, y2, x2))
	# 아래서부턴 positive_count + negative_count를 rois_count로 표현하겠음
	rois = tf.concat([positive_rois, negative_rois], axis=0)
	N = tf.shape(negative_rois)[0] 	# negative_count
	
	# rois.shape[0]이 최소한 TRAIN_ROIS_PER_IMAGE가 될 수 있도록 1차원에만 zeropadding
	P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
	rois = tf.pad(rois, paddings = [(0, P), (0, 0)])
	
	# roi_gt_boxes.shape[0]의 최소값 TRAIN_ROIS_PER_IMAGE. 위와 같은 방식
	# roi_gt_boxes.shape = (rois_count, (y1, x1, y2, x2))
	roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
	# roi_gt_class_ids.shape = (rois_count)		위와 같은 방식
	roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
	# deltas.shape = (rois_count, (y1, x1, y2, x2))
	deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
	# masks.shape (rois_count, MASK_SHAPE[0], MASK_SHAPE[1])
	masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])
	
	return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KL.Layer):
	"""
	Subsamples proposals and generates target box refinement, class_ids,
	and masks for each.
	"""
	def __init__(self, config, **kwargs):
		super(DetectionTargetLayer, self).__init__(**kwargs)
		self.config = config
        
	def call(self, inputs):
		"""
		Inputs: [target_rois, input_gt_class_ids, gt_boxes, input_gt_masks]
			target_rois : [batch, num_rois, (y1, x1, y2, x2)]
					in normalized coordinates.
					
			input_gt_class_ids : [batch, MAX_GT_INSTANCES], 	Integer class IDs. 	
			gt_boxes : [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
					in normalized coordinates.
			input_gt_masks : [batch, height, width, MAX_GT_INSTANCES]
					of boolean type
			
		Returns: Target ROIs and corresponding class IDs, bounding box shifts, and masks.
			rois: [batch, rois_count, (y1, x1, y2, x2)] 
					in normalized coordinates
			target_class_ids:	[batch, rois_count], Integer class IDs.
			target_deltas:	[batch, rois_count, (dy, dx, dh, dw]
			target_mask:	[batch, rois_count, height, width]
					Masks cropped to bbox boundaries and resized to neural
					network output size.
					TRAIN_ROIS_PER_IMAGE =< rois_count
				
		Note: Returned arrays might be zero padded if not enough target ROIs.
		"""
		proposals = inputs[0]		# proposals == target_rois
		gt_class_ids = inputs[1]	# gt_class_ids == input_gt_class_ids
		gt_boxes = inputs[2]		# gt_boxes == gt_boxes
		gt_masks = inputs[3]		# gt_masks == input_gt_masks
		
		# Slice the batch and run a graph for each slice
		names = ["rois", "target_class_ids", "target_deltas", "target_mask"]
		outputs = utils.batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],
									lambda w, x, y, z: detection_targets_graph(
										w, x, y, z, self.config),
									self.config.IMAGES_PER_GPU, names=names)
			
		return outputs


def parse_image_meta_graph(meta):
	"""
	Parses a tensor that contains image attributes to its components.
	See compose_image_meta() for more details.
	meta: [batch, meta length] where meta length depends on NUM_CLASSES
	Returns a dict of the parsed tensors.
	"""
	image_id = meta[:, 0]
	original_image_shape = meta[:, 1:4]
	image_shape = meta[:, 4:7]
	window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
	scale = meta[:, 11]
	active_class_ids = meta[:, 12:]
	return {
			"image_id": image_id,
			"original_image_shape": original_image_shape,
			"image_shape": image_shape,
			"window": window,
			"scale": scale,
			"active_class_ids": active_class_ids,
    }


class ProposalLayer(KL.Layer):
	"""
	1. Receives anchor scores and selects a subset to pass as proposals to the second stage.
	2. Filtering is done based on anchor scores and non-max suppression to remove overlaps. (overlaps = iou)
	3. anchor에 delta 적용
	"""
		
	def __init__(self, config, proposal_count, nms_threshold, **kwargs):
		super(ProposalLayer, self).__init__(**kwargs)
		"""
		input : 
			proposal_count : num of proposal region, 
				training : 2000, inference : 1000
			nms_threshold : 0.7
		"""
		self.config = config
		self.proposal_count = proposal_count
		self.nms_threshold = nms_threshold
        
	def call(self, inputs):
		"""
		Inputs: [rpn_probs, rpn_bbox, anchors]
			rpn_probs : [batch, h * w * anchors_per_location, [background_prob, foreground_prob]] 
			rpn_bbox : [batch, h * w * anchors_per_location, [dx, dy, dw, dh]]
			anchors : [batch, h * w * anchors_per_location, (y1, x1, y2, x2)] 
					anchors in normalized coordinates
				
		Returns:
			Proposals : [batch, rois, (y1, x1, y2, x2)]
				Proposals in normalized coordinates 
		"""
		# Box Scores. Use the foreground class confidence. [batch, num_anchors]
		scores = inputs[0][:, :, 1]
        
		# Box deltas.shape = (batch, num_rois, 4)
		deltas = inputs[1]
		# multiply refinement standard deviation
		deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        
		# Anchors
		anchors = inputs[2]
        
		# 상위foreground_prob 기준, pre_nms_limit개수를 제외하고 버리는 방식으로 성능 향상
		pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
		# ix : indices of the pre_nms_limit top value 
		# ix.shape = (batch, pre_nms_limit)
		ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
						 name="top_anchors").indices
        
		# scores.shape = ( 1, (score_batch_1, score_batch_2))
		# 각 score는 pre_nms_limit개수의 상위 prob를 담고있다
		scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
									self.config.IMAGES_PER_GPU)
		# deltas.shape = (1, (deltas_batch_1, deltas_batch_2))
		deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
									self.config.IMAGES_PER_GPU)
		# pre_nms_anchors.shape = (1, (anchors_batch_1, anchors_batch_2))
		pre_nms_anchors = utils.batch_slice([anchors, ix],
											lambda a, x: tf.gather(a, x),
											self.config.IMAGES_PER_GPU,
											names=["pre_nms_anchors"])
        
		# Apply deltas to anchors to get refined anchors.
		# boxes.shape = (batch, N, (y1, x1, y2, x2))
		boxes = utils.batch_slice([pre_nms_anchors, deltas],
								  lambda x, y: utils.apply_box_deltas_graph(x, y),
								  self.config.IMAGES_PER_GPU,
								  names=["refined_anchors"])
		# Clip to image boundaries. 
		# Since we're in normalized coordinates, clip to 0..1 range. 
		# boxes.shape = (batch, N, (y1, x1, y2, x2))
		window = np.array([0, 0, 1, 1], dtype=np.float32)
		boxes = utils.batch_slice(boxes,
								  lambda x: utils.clip_boxes_graph(x, window),
								  self.config.IMAGES_PER_GPU,
								  names=["refined_anchors_clipped"])
				
		# Non-max suppression
		def nms(boxes, scores):
		# iou_threshold = default 0.5
			indices = tf.image.non_max_suppression(
					boxes, scores, self.proposal_count,
					score_threshold = self.nms_threshold,
					name="rpn_non_max_suppression")
			proposals = tf.gather(boxes, indices)
            
			# zero pad if needed
			padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
			proposals = tf.pad(proposals, [(0, padding), (0, 0)])
			return proposals
		# proposals.shape = (batch, num_rois, (y1, x1, y2, x2))
		proposals = utils.batch_slice([boxes, scores], nms,
									   self.config.IMAGES_PER_GPU)
        
		return proposals


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
	"""
	Builds the computation graph of Region Proposal Network.
	
	feature_map: backbone features [batch, height, width, depth]
	anchors_per_location: number of anchors per pixel in the feature map
	anchor_stride: If 1 anchors are created for each cell in the backbone feature map.
		
	Return:
		rpn_class_logits: [batch, h * w * anchors_per_location, [background_logit, foreground_logit]] 
					Anchor classifier logits (before softmax)
		rpn_probs: [batch, h * w * anchors_per_location, [background_prob, foreground_prob]] 
					Anchor classifier probabilities.
		rpn_bbox: [batch, h * w * anchors_per_location, [dx, dy, dw, dh]]
					Deltas to be applied to anchors.	
	"""
    
	## Intermediate layer
	shared = KL.Conv2D(filters =512, kernel_size = 3, padding='same', 
                    	strides = anchor_stride, name='rpn_conv_shared', 
                    	activation='relu')(feature_map)
        
	## cls layer
	# Anchor Score. [batch, height, width, anchors per location * 2].
	x = KL.Conv2D(filters = 2*anchors_per_location, kernel_size = 1, padding='valid', 
					name='rpn_class_raw', activation='linear')(shared)
	   
	# Reshape from [batch, anchors*2] to [batch, anchors, 2]
	# [batch, h * w * anchors_per_location, [background_logit, foreground_logit]]
	rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    
	# Softmax on last dimension of BG/FG.
	# probability about background or foreground
	# [batch, h * w * anchors_per_location, [background, foreground]]
	rpn_probs = KL.Activation('softmax', name="rpn_class_xxx")(rpn_class_logits)
        
	## reg layer
	# Bounding box refinement. [batch, H, W, anchors per location * depth]
	# depth is [x, y, log(w), log(h)]
	x = KL.Conv2D(filters = 4*anchors_per_location, kernel_size = 1, padding='valid', 
                   name='rpn_bbox_pred', activation='linear')(shared)
        
	# Reshape from [batch, anchors*4] to [batch, anchors, 4]
	# [batch, h * w * anchors_per_location, [dx, dy, dw, dh]]
	rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
        
	output = [rpn_class_logits, rpn_probs, rpn_bbox]  
	return output

def build_rpn_model(anchor_stride, anchors_per_location, depth):
	"""
	Builds a Keras model of the Region Proposal Network.
	shared weights로 여러 번 사용할 수 있도록 rapping
		
	anchor_stride: If 1 anchors are created for each cell in the backbone feature map.
	anchors_per_location: number of anchors per pixel in the feature map
	depth: Depth of the backbone feature map.
			
	Returns a Keras Model object. The model outputs, when called, are:
		rpn_class_logits: [batch, h * w * anchors_per_location, [background_logit, foreground_logit]] 
					Anchor classifier logits (before softmax)
		rpn_probs: [batch, h * w * anchors_per_location, [background_prob, foreground_prob]] 
 					Anchor classifier probabilities.
		rpn_bbox: [batch, h * w * anchors_per_location, [dx, dy, dw, dh]]
					Deltas to be applied to anchors.				
	"""
    
	input_feature_map = KL.Input(shape=[None, None, depth],
								  name="input_rpn_feature_map")
		
	# Builds the computation graph of RPN
	# output = [rpn_class_logits, rpn_probs, rpn_bbox]
	outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
	return KL.Model([input_feature_map], outputs, name="rpn_model")


def conv_block(input_tensor, filters, strides = 2, 
			   stage = 0, block = None, use_bias=True, train_bn=True):
	"""
	input_tensor : input
	filters : [nb_filter1, nb_filter2, nb_filter3]
	stage : stage number
	block : block name
	train_bn : whether layer freeze
	"""
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
    
	nb_filter1, nb_filter2, nb_filter3 = filters
    
	x = KL.Conv2D(filters = nb_filter1, kernel_size = 1, strides = strides,
                   name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
	x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.Conv2D(filters = nb_filter2, kernel_size = 3, padding='same', 
				   name=conv_name_base + '2b', use_bias=use_bias)(x)
	x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
	x = KL.Activation('relu')(x)     
    
	x = KL.Conv2D(filters = nb_filter3, kernel_size = 1, 
               	   name=conv_name_base + '2c', use_bias=use_bias)(x)
	x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)
    
	shortcut = KL.Conv2D(filters = nb_filter3, kernel_size = 1, strides=strides,
                          name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
	shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)
    
	x = KL.Add()([x, shortcut])
	x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x) 
    
	return x    

def identity_block(input_tensor, filters, stage, block, use_bias=True, train_bn=True):
	"""
	input_tensor : input
	filters : [nb_filter1, nb_filter2, nb_filter3]
	stage : stage number
	block : block name
	train_bn : whether layer freeze 
	"""
	nb_filter1, nb_filter2, nb_filter3 = filters
    
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
                                                  
	x = KL.Conv2D(filters = nb_filter1, kernel_size = 1, 
				   name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
	x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
	x = KL.Activation('relu')(x)
    
	x = KL.Conv2D(filters = nb_filter2, kernel_size = 3, padding='same',
 				   name=conv_name_base + '2b', use_bias=use_bias)(x)
	x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
	x = KL.Activation('relu')(x)              
    
	x = KL.Conv2D(filters = nb_filter3, kernel_size = 1, 
				   name=conv_name_base + '2c', use_bias=use_bias)(x)
	x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

	x = KL.Add()([x, input_tensor])
	x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x) 
	return x

def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
	"""
	input_image : input으로 받은 순수 image
	architecture : "resnet50" or "resnet101"
	stage5 : whether use stage5 
	train_bn : whether layer freeze 
	"""
	assert architecture in ["resnet50", "resnet101"]
    
    # Stage 1
	x = KL.ZeroPadding2D(padding = 3)(input_image)
	x = KL.Conv2D(filters =64, kernel_size = 7, strides = 2, name='conv1')(x)
	x = KL.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, name='bn_conv1')(x, training = train_bn)
	x = KL.Activation('relu')(x) 
	x = KL.MaxPooling2D(poolsize = 3, strides = 2, padding="same")(x)
	C1 = x
    
	# Stage 2
	x = conv_block(x, [64, 64, 256], strides = 1, stage=2, block='a', train_bn = train_bn)
	x = identity_block(x, [64, 64, 256], stage=2, block='b', train_bn = train_bn)
	x = identity_block(x, [64, 64, 256], stage=2, block='c', train_bn = train_bn)
	C2 = x
    
	# Stage 3
	x = conv_block(x, [128, 128, 512], stage=3, block='a', train_bn = train_bn)
	x = identity_block(x, [128, 128, 512], stage=3, block='b', train_bn = train_bn)
	x = identity_block(x, [128, 128, 512], stage=3, block='c', train_bn = train_bn)
	x = identity_block(x, [128, 128, 512], stage=3, block='d', train_bn = train_bn)
	C3 = x
    
	# Stage 4
	x = conv_block(x, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
	block_count = {"resnet50": 5, "resnet101": 22}[architecture]
	for i in range(block_count):
		x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
	C4 = x
    
	# Stage 5
	if stage5 : 
		x = conv_block(x, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
		x = identity_block(x, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
		x = identity_block(x, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
		C5 = x
	else :
		C5 = None
	return [C1, C2, C3, C4, C5]

def fpn_graph(input_image, config):
	# Build the shared convolutional layers.
	# Bottom-up Layers
	# Returns a list of the last layers of each stage, 5 in total.
	# Don't create the thead (stage 5), so we pick the 4th item in the list.
    
	_, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                             stage5=True, train_bn=config.TRAIN_BN)
    
	# Top-down Layers
	# TODO: add assert to varify feature map sizes match what's in config
	M5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    
	M4 = KL.Add(name="fpn_p4add")([
			KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(M5),
			KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    
	M3 = KL.Add(name="fpn_p3add")([
			KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(M4),
			KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    
	M2 = KL.Add(name="fpn_p2add")([
			KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(M3),
			KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    
    # Attach 3x3 conv to all P layers to get the final feature maps.
	P2 = KL.Conv2D(config.TOP_DOWN_LAST_FILTER, (3, 3), padding="SAME", name="fpn_p2")(M2)
	P3 = KL.Conv2D(config.TOP_DOWN_LAST_FILTER, (3, 3), padding="SAME", name="fpn_p3")(M3)
	P4 = KL.Conv2D(config.TOP_DOWN_LAST_FILTER, (3, 3), padding="SAME", name="fpn_p4")(M4)
	P5 = KL.Conv2D(config.TOP_DOWN_LAST_FILTER, (3, 3), padding="SAME", name="fpn_p5")(M5)
    
	# P6 is used for the 5th anchor scale in RPN. Generated by
	# subsampling from P5 with stride of 2.
	P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    
	# Note that P6 is used in RPN, but not in the classifier heads.
	rpn_feature_maps = [P2, P3, P4, P5, P6]
	mrcnn_feature_maps = [P2, P3, P4, P5]
    
	return rpn_feature_maps, mrcnn_feature_maps


def load_image_gt(dataset, image_id, config):
	"""
	dataset : train dataset or validation dataset 중에서 1 장의 image에 대한 data
    
	image_id : index of image

	Returns:
	image: [height, width, 3]
	shape: the original shape of the image before resizing and cropping.
	class_ids: [instance_count] Integer class IDs
	bbox: [instance_count, (y1, x1, y2, x2)]
	mask: [height, width, instance_count]. The height and width are GT_Bbox height and width
	"""

	## image
	image = dataset["image"]["original_image"]  	# (256, 256, 3)

	## mask, class_ids
	mask_list = list()
	class_ids_list = list()
	instance_count = len(dataset["annotation"])
	for i in range(instance_count) :
		height_GTBox = dataset["annotation"][i]["height"]
		width_GTBox = dataset["annotation"][i]["width"]
		# mask.shape = (height, width, 1)
		mask = [height_GTBox, width_GTBox, dataset["annotation"][i]["class_id"]]
		mask_list.append(mask)

		class_ids = dataset["annotation"][i]["class_id"]
		class_ids_list.append(class_ids)
	
	# mask.shape = (height, width, instance_count)
	mask = np.dstack([mask_list[j] for j in range(len(dataset["annotation"]))])
	# class_ids.shape = (instance_count, )  value =  [0, 1]
	class_ids = [class_ids_list[j] for j in range(len(dataset["annotation"]))]

	## original_shape
	original_shape = image.shape

	## image.shape = (height, width, 3)
	# window.shape = (y1, x1, y2, x2)
	# scale : The scale factor used to resize the image
	# padding.shape = (2, 2, 2) : ((top, bottom), (left, right), (0, 0))
	image, window, scale, padding, crop = utils.resize_image(image, 
											mode = config.MODE,
											min_dim=config.IMAGE_MIN_DIM,
											min_scale=config.IMAGE_MIN_SCALE,
											max_dim=config.IMAGE_MAX_DIM)

	# mask도 resize후 zeropadding
	# mask.shape = [height, width, instance_count]
	mask = utils.resize_mask(mask, scale, padding, crop)	

	# 몇몇 mask는 resize된 과정에서 아예 0의 값을 가진 경우가 있음. 이런 mask는 걸러낸다.
	# 해당 project에선 해당 없을것임
	_idx = np.sum(mask, axis=(0, 1)) > 0
	mask = mask[:, :, _idx]
	class_ids = class_ids[_idx]

	## Active classes
	# class의 개수 중 실제 dataset에서 사용되는 class에는 1을 할당
	# 해당 프로젝트에선 모든 class가 사용되므로 전부 1
	num_classes = len(class_ids)
	active_class_ids = np.ones(num_classes, dtype=np.int32)

	## Bbox
	bbox_list = list()
	for i in range(len(dataset["annotation"])) :
		# Bbox [1, (y1, x1, y2, x2)]
		bbox = [dataset["annotation"][i]["class_id"],
				dataset["annotation"][i]["bbox"]]
		bbox_list.append(bbox)
	# Bbox.shape = (num_instances, 4),	  4: (y1, x1, y2, x2)
	bbox = np.hstack([bbox_list[j] for j in range(len(dataset["annotation"]))])

	# Image meta data
	image_meta = utils.compose_image_meta(image_id, original_shape, image.shape,
									window, scale, active_class_ids)

	return image, image_meta, class_ids, bbox, mask


def build_rpn_targets(config, anchors, gt_boxes):
	"""
	compute overlaps and identify positive anchors and deltas 
	to refine them to match their corresponding GT boxes.
	anchors : [anchor_count, (y1, x1, y2, x2)]
	gt_boxes : [instance_count, (y1, x1, y2, x2)]
    
	Returns:
		rpn_match: [N] (int32) matches between anchors and GT boxes.
			N : 1 = positive anchor, -1 = negative anchor, 0 = neutral
		rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    	
	Notice: A crowd box are ignored.
	"""

	# RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
	# rpn_match.shape = (anchor_count)
	rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)

	# rpn_bbox.shape = (max anchors per image, 4), 		4 : [dy, dx, dh, dw]
	rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

	# Compute overlaps (IOU를 계산)
	# overlaps.shape = (num_anchors,  num_IOU)  , num_IOU = num_gt_boxes
	overlaps = utils.compute_overlaps(anchors, gt_boxes)

	## Match anchors to GT Boxes
	# If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
	# If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
	# Neutral anchors don't influence the loss function.
	## However, don't keep any GT box unmatched (rare, but happens).
	# Instead, match it to the closest anchor (even if its max IoU is < 0.3).
    #
	# 1. Set negative anchors first. 
	# They get overwritten below if a GT box is matched to them.
	# overlaps에서 0차원 방향으로 slicing
	anchor_iou_argmax = np.argmax(overlaps, axis=1)
	anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
	rpn_match[(anchor_iou_max < 0.3)] = -1

	# 2. Set an anchor for each GT box (regardless of IoU value).
	# If multiple anchors have the same IoU match all of them
	# np.max(overlaps, axis=0) : [num_anchors], 
	# gt_iou_argmax : where about max iou anchor in from each gt_boxes
	# overlaps에서 1차원 방향으로 slicing
	gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
	rpn_match[gt_iou_argmax] = 1

	# 3. Set anchors with high overlap as positive.
	rpn_match[anchor_iou_max >= 0.7] = 1
	# 예시 anchor_count = 4, instance_count = 2
	# [[0.1  0.8]			# max > 0.7			1
	# [0.2  0.5]        	# 3 < max < 0.7		0
	# [0.25 0.9 ]			# max > 0.7			1
	# [0.1  0.2]]			# max < 0.3			-1
	# rpn_match : [ 1  0  1 -1]

	## Subsample to balance positive and negative anchors
	# Don't let positives be more than half the anchors
	# positive : negative == 1 : 1
	ids = np.where(rpn_match == 1)[0]
	extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
	if extra > 0: 
		# positives anchor중에서 (RPN_TRAIN_ANCHORS_PER_IMAGE // 2) 개를 제외하곤 다 버림
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0

	# Same for negative proposals
	ids = np.where(rpn_match == -1)[0]
	extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
	if extra > 0:
	# negative anchor중에서 (RPN_TRAIN_ANCHORS_PER_IMAGE // 2) 개를 제외하곤 다 버림
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0

	## For positive anchors, compute shift and scale needed to transform them
	# to match the corresponding GT boxes.
	ids = np.where(rpn_match == 1)[0]
	ix = 0  # index into rpn_bbox

	for i, a in zip(ids, anchors[ids]):
		# Closest gt box (it might have IoU < 0.7)
		gt = gt_boxes[anchor_iou_argmax[i]]
        
		# Convert coordinates to center plus width/height.
		# GT Box
		gt_h = gt[2] - gt[0]
		gt_w = gt[3] - gt[1]
		gt_center_y = gt[0] + 0.5 * gt_h
		gt_center_x = gt[1] + 0.5 * gt_w
		# Anchor
		a_h = a[2] - a[0]
		a_w = a[3] - a[1]
		a_center_y = a[0] + 0.5 * a_h
		a_center_x = a[1] + 0.5 * a_w

		# Compute the bbox refinement that the RPN should predict.
		rpn_bbox[ix] = [
			(gt_center_y - a_center_y) / a_h,
			(gt_center_x - a_center_x) / a_w,
			np.log(gt_h / a_h),
			np.log(gt_w / a_w),
		]
		# Normalize
		rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
		ix += 1
	
	return rpn_match, rpn_bbox


# random_rois 부분 사용되는지 확인 후 추가하기
def data_generator(dataset, config, shuffle=True,
                   random_rois=0, batch_size=1):
	"""
	dataset : train data or validation data
	shuffle : whether shuffle
	batch_size : How many images to return in each call
	random_rois : If > 0 then generate proposals to be used to train the
                 network classifier and mask heads.
                 Useful if training the Mask RCNN part without the RPN.
	detection_targets : If True, generate detection targets 
    					(class IDs, bbox, deltas, and masks).
    					
	inputs list:
	- images: [batch, H, W, C]
	- image_meta: [batch, (meta data)] Image details. See compose_image_meta()
	- rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
	- rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	- gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
	- gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
	- gt_masks: [batch, height, width, MAX_GT_INSTANCES]
		The height and width are those of the image unless use_mini_mask is True, 
		in which case they are defined in MINI_MASK_SHAPE.
	"""

	# dataset.keys() : "annotation", "image"
	
	batch_item_index = 0  # batch item index
	image_index = -1
	image_ids = [i for i in len(dataset)]		# [0, 1, ...]
	error_count = 0

	## Anchors
	backbone_shapes = utils.compute_backbone_shapes(config, config.IMAGE_SHAPE)
	# anchors.shape = (anchor_count, 4), 	4 : (y1, x1, y2, x2)
	anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, 
									   		 config.RPN_ANCHOR_RATIOS,
									   		 backbone_shapes,
									   		 config.BACKBONE_STRIDES,
									    	 config.RPN_ANCHOR_STRIDE)

	while True:		
		try:	
			# Increment index to pick next image.	
			image_index = (image_index + 1) % len(image_ids)

			# Shuffle if at the start of an epoch
			if shuffle and image_index == 0:
				np.random.shuffle(image_ids)

			# 현재 image index
			image_id = image_ids[image_index]

			# Get GT bounding boxes and masks for image.
			# image.shape = (height, width, 3)  resized
			# gt_class_ids.shape =  (instance_count) == [0, 1]
			# gt_boxes.shape = (instance_count, 4),  	4 : [y1, x1, y2, x2]
			# gt_masks.shape = (height, width, instance_count)
			# instance_count : number of objects which in image
			(image, image_meta, gt_class_ids, 
			gt_boxes, gt_masks) = load_image_gt(dataset, image_id, config)

			# Skip images that have no instances.
			# 해당 프로젝트의 data에선 해당없음 
			if not np.any(gt_class_ids > 0):
				continue

			# RPN Targets
			# rpn_match: [N] 		e.g. [1, 0, 1, -1, ...]
			#	N : 1 = positive anchor, -1 = negative anchor, 0 = neutral
			# 	
			# rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
			rpn_match, rpn_bbox = build_rpn_targets(config, anchors, gt_class_ids, gt_boxes)

			## Mask R-CNN Targets
			# Init batch arrays
			if batch_item_index == 0:
				batch_image_meta = np.zeros((batch_size, image_meta.shape[0]), 
											 dtype=image_meta.dtype)
				batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1],
											dtype=rpn_match.dtype)
				batch_rpn_bbox = np.zeros([batch_size, 
										   config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4],
										   dtype=rpn_bbox.dtype)
										   
				batch_images = np.zeros((batch_size,) + image.shape,
										 dtype=np.float32)
				batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES),
											   dtype=np.int32)
				batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4),
										   dtype=np.int32)
				batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1], 
										   config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)

			# If more instances than fits in the array, sub-sample from them.
			# 해당 프로젝트의 data에선 해당없을것임
			if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
				ids = np.random.choice(np.arange(gt_boxes.shape[0]),
									   config.MAX_GT_INSTANCES, replace=False)
				gt_class_ids = gt_class_ids[ids]
				gt_boxes = gt_boxes[ids]
				gt_masks = gt_masks[:, :, ids]

			# Add to batch
			# batch_size중 batch_item_index번째에 각각의 data 할당 
			batch_image_meta[batch_item_index] = image_meta
			batch_rpn_match[batch_item_index] = rpn_match[:, np.newaxis]	# shape맞춰주기
			batch_rpn_bbox[batch_item_index] = rpn_bbox	# Delta
			batch_images[batch_item_index] = image.astype(np.float32)
			batch_gt_class_ids[batch_item_index, :gt_class_ids.shape[0]] = gt_class_ids
			batch_gt_boxes[batch_item_index, :gt_boxes.shape[0]] = gt_boxes
			batch_gt_masks[batch_item_index, :, :, :gt_masks.shape[-1]] = gt_masks
            
			# Batch full인 경우
			if batch_item_index >= batch_size:
				inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
						  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
				outputs = list()

				# yield를 이용해서 inputs을 return하고 while문은 다시 continue
				yield inputs
                
				# start a new batch
				batch_item_index = 0

		except (GeneratorExit, KeyboardInterrupt):
			raise
		except:
			# Log it and skip the image
			logging.exception(f"Error processing image {image_id[image_index]}")
			error_count += 1
			if error_count > 5:
				raise
									 


class MaskRCNN(KM.Model):
	def __init__(self, mode, config, model_dir):
		"""
		mode : "training" or "inference"
		model_dir : path
		"""
		super(MaskRCNN, self).__init__()
		self.mode = mode
		self.config = config
		self.model_dir = model_dir
		self.set_log_dir()
		self.keras_model = self.build(mode=mode) 

	
	def set_log_dir(self, model_path=None):
		self.epoch = 0
		now = datetime.datetime.now()		# 최초 model 학습 시작 년월일시분
		
		if model_path: # model_path가 존재한다 == 이전에 이미 학습중인 model이 존재했다.
			# Continue from we left of. Get epoch and date from the file name
			# model_path = "model_dir/20210901T1644/mask_rcnn_lung_0001.h5"
			regex = r".*[/\\]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
			m = re.match(regex, model_path)
			if m:
				# 년월일시분, epoch 재정의 
				now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
											int(m.group(4)), int(m.group(5)))
				self.epoch = int(m.group(6))
				print(f'Re-starting from epoch {self.epoch}')
		
		# Directory for training logs
		tmp = str(now)
		ymd, time = tmp.split(' ')
		y, m, d = ymd.split('-')
		hour, min, _ = time.split(':')
		self.log_dir = os.path.join(self.model_dir, f"{y}{m}{d}T{hour}{min}")
		# model_dir/20210901T0806
		
		self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_lung_"+ f"{self.epoch:04d}.h5")
		# model_dir/20210901T0815/mask_rcnn_lung_0001.h5


	def build(self, mode):
		"""
		Build Mask R-CNN architecture.
		input_shape: The shape of the input image.
		"""
		
		assert mode in ['training', 'inference']
		
		# Image size must be dividable by 2 multiple times
		h, w = self.config.IMAGE_SHAPE[:2]
		if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
			raise Exception("Image size must be dividable by 2 at least 6 times "
								"to avoid fractions when downscaling and upscaling."
								"For example, use 256, 320, 384, 448, 512, ... etc. ")

		## Inputs
		# input_image.shape = (batch, width, height, 3)
		input_image = KL.Input(shape=[None, None, 3], name="input_image")
		# input_image_meta.shape = (batch, None == IMAGE_META_SIZE)
		input_image_meta = KL.Input(shape=([self.config.IMAGE_META_SIZE]),
									 name="input_image_meta")

		if mode == "training":
			## RPN GT
			# input_rpn_match.shape = (batch, anchors, 1)
			input_rpn_match = KL.Input(shape=[None, 1], 
									name="input_rpn_match", dtype=tf.int32)
			# input_rpn_bbox.shape = (batch, max positive anchors, (dy, dx, dh, dw))
			input_rpn_bbox = KL.Input(shape=[None, 4], 
								name="input_rpn_bbox", dtype=tf.float32)
			
			## Detection GT (class IDs, bounding boxes, and masks)
			# 1. GT Class IDs (zero padded)
			# input_gt_class_ids.shape = (batch, MAX_GT_INSTANCES)
			input_gt_class_ids = KL.Input(shape=[None],
									name="input_gt_class_ids", dtype=tf.int32)
			
			# 2. GT Boxes in pixels (zero padded)
			# input_gt_boxes.shape = (batch, MAX_GT_INSTANCES, (y1, x1, y2, x2))
			# in image coordinates
			input_gt_boxes = KL.Input(shape=[None, 4], 
								name="input_gt_boxes", dtype=tf.float32)

			
			# Normalize coordinates of gt_boxes
			# gt_boxes.shape = (batch, MAX_GT_INSTANCES, (y1, x1, y2, x2))   
			gt_boxes = KL.Lambda(lambda x: utils.norm_boxes_graph2(x))([input_gt_boxes, input_image])
			
			# 3. GT Masks (zero padded)
			# input_gt_masks.shape = (batch, height, width, MAX_GT_INSTANCES)
			input_gt_masks = KL.Input(
						shape=[self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], None],
						name="input_gt_masks", dtype=bool)

		elif mode == "inference":
			# Anchors in normalized coordinates
				input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

		## get feature map from FPN
		# rpn_feature_maps = [P2, P3, P4, P5, P6]
		# mrcnn_feature_maps = [P2, P3, P4, P5]
		rpn_feature_maps, mrcnn_feature_maps = fpn_graph(input_image, self.config)
		print("fpn_graph 통과")																###

		## Anchors
		if mode == "training":
		# anchors.shape = (anchor_counts, (y1, x1, y2, x2))
		# y1, x1, y2, x2 has normalized
			anchors = self.get_anchors(self.config.IMAGE_SHAPE)	
			
			# Duplicate across the batch dimension because Keras requires it
			# BATCH_SIZE = 3이면 3배 늘린다.
			anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
			
			# A hack to get around Keras's bad support for constants
			anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)		
		else:
			anchors = input_anchors
				
		## RPN Model
		rpn = build_rpn_model(self.config.RPN_ANCHOR_STRIDE,
							len(self.config.RPN_ANCHOR_RATIOS), self.config.TOP_DOWN_PYRAMID_SIZE)
		print("build_rpn_model 통과")																###

		# Loop through pyramid layers
		layer_outputs = list()  # list of lists
		for p in rpn_feature_maps:
			layer_outputs.append(rpn([p]))
			# [[rpn_class_logits, rpn_probs, rpn_bbox], ...] P2부터 P6까지 5회


		# Concatenate layer outputs
		# e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
		# [[rpn_class_logits_p2, rpn_class_logits_p3, ...], [rpn_probs_p1, rpn_probs_p2, ...]]
		output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
		outputs = list(zip(*layer_outputs))
		outputs = [KL.concatenate(axis=1, name=n)(list(o))
					for o, n in zip(outputs, output_names)]

		# rpn_class_logits.shape = (batch, anchors, (background_logit, foreground_logit))
		# rpn_class.shape = (batch, anchors, (background_prob, foreground_prob))
		# rpn_probs_p1.shape = (batch, anchors, (background_prob, foreground_prob))
		# rpn_bbox_p1.shape =  (batch, anchors, (dx, dy, dw, dh))
		# anchors.shape = h * w * anchors_per_location * 4 (P2~P6)
		rpn_class_logits, rpn_class, rpn_bbox = outputs

		# set proposals count for generate proposals
		if mode == "training" :
			proposal_count = self.config.POST_NMS_ROIS_TRAINING 	# 2000
		else :
			proposal_count = self.config.POST_NMS_ROIS_INFERENCE	# 1000

		# Generate proposals
		# Proposals are [batch, num_rois, (y1, x1, y2, x2)] in normalized coordinates
		# and zero padded.
		rpn_rois = ProposalLayer(self.config,
					proposal_count = proposal_count,
					nms_threshold = self.config.RPN_NMS_THRESHOLD  ,
					name="ROI" )([rpn_class, rpn_bbox, anchors])

		if mode == "training":
			# Class ID를 표시하기 위한 Class ID mask
			# active_class_ids.shape = (batch, NUM_CLASSES).  value = 1
			active_class_ids = KL.Lambda(
					lambda x: parse_image_meta_graph(x)["active_class_ids"]
					)(input_image_meta)

			if not self.config.USE_RPN_ROIS:
				# Ignore predicted ROIs and use ROIs provided as an input.
				input_rois = KL.Input(shape=[self.config.POST_NMS_ROIS_TRAINING, 4],
										name="input_roi", dtype=np.int32)
				# Normalize coordinates
				target_rois = KL.Lambda(lambda x: utils.norm_boxes_graph(
						x, tf.shape(input_image)[1:3]))(input_rois)
			else : 	# 해당됨
				target_rois = rpn_rois

			## Generate detection targets
			# Subsamples proposals and generates target outputs for training
			# Note that proposal class IDs, gt_boxes, and gt_masks are zero padded where 2-D(rois_count).
			# Equally, returned rois and targets are zero padded.
			
			# rois.shape = (batch, num_rois, (y1, x1, y2, x2)) 
			# target_class_ids.shape = (batch, num_rois), Integer class IDs.
			# target_bbox.shape = (batch, num_rois, (dy, dx, dh, dw))
			# target_mask.shape = (batch, num_rois, height, width)
			# 	Masks cropped to bbox boundaries and resized to neural network output size.
			# 	TRAIN_ROIS_PER_IMAGE =< num_rois
			rois, target_class_ids, target_bbox, target_mask =\
					DetectionTargetLayer(self.config, name="proposal_targets")([
						target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

			## Network Heads
			# TODO: verify that this handles zero padded ROIs
			# mrcnn_class_logits.shape = (batch, num_rois, NUM_CLASSES) classifier logits (before softmax)
			# mrcnn_class.shape = (batch, num_rois, NUM_CLASSES) classifier probabilities
			# mrcnn_bbox.shape = (batch, num_rois, NUM_CLASSES, (dy, dx, dh, dw)) Deltas to apply to
			mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(self.config,
									rois, mrcnn_feature_maps, input_image_meta,
									self.config.POOL_SIZE, self.config.NUM_CLASSES, 
									train_bn=self.config.TRAIN_BN,
									fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)

			## mrcnn_mask.shape = (batch, num_rois, (MASK_POOL_SIZE - 1)*2 + 2, (MASK_POOL_SIZE - 1)*2 + 2, num_classes)
			# == [batch, num_rois, h, w, num_classes]
			mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
												input_image_meta,
												self.config.NUM_CLASSES,
												self.config.MASK_POOL_SIZE,
												train_bn=self.config.TRAIN_BN)

			output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

			## Losses
			rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x),
									name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
			rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(self.config, *x), name="rpn_bbox_loss")(
					[input_rpn_bbox, input_rpn_match, rpn_bbox])
			class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
					[target_class_ids, mrcnn_class_logits, active_class_ids])
			bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
					[target_bbox, target_class_ids, mrcnn_bbox])
		
			mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
					[target_mask, target_class_ids, mrcnn_mask])

			## Model
			inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
					input_gt_class_ids, input_gt_boxes, input_gt_masks]

			if not self.config.USE_RPN_ROIS:	# 해당없음
				inputs.append(input_rois)

			outputs = [rpn_class_logits, rpn_class, rpn_bbox,
					mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
					rpn_rois, output_rois,
					rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
			model = KM.Model(inputs, outputs, name='mask_rcnn') 
		
		else: 	# mode == "inference"
			## Network Heads
			# Proposal classifier and BBox regressor heads
			# mrcnn_class_logits =  (batch, num_rois, NUM_CLASSES) classifier logits (before softmax)
			# mrcnn_class.shape = (batch, num_rois, NUM_CLASSES) classifier probabilities
			# mrcnn_bbox = (batch, num_rois, NUM_CLASSES, (dy, dx, dh, dw)) 
			#	Deltas to apply to proposal boxes
			mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
									rpn_rois, mrcnn_feature_maps, input_image_meta,
									self.config.POOL_SIZE, self.config.NUM_CLASSES, 
									train_bn=self.cinfig.TRAIN_BN,
									fc_layers_size = self.config.FPN_CLASSIF_FC_LAYERS_SIZE)

			## Detections
			# output is [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] 
			# in normalized coordinates
			detections = DetectionLayer(self.config, name="mrcnn_detection")(
					[rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

			# Create masks for detections
			# detection_boxes : [batch, num_detections, (y1, x1, y2, x2)] 
			detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
			mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
												input_image_meta,
												self.config.MASK_POOL_SIZE,
												self.config.NUM_CLASSES,
												train_bn = self.config.TRAIN_BN)
			
			model = KM.Model([input_image, input_image_meta, input_anchors],
								[detections, mrcnn_class, mrcnn_bbox,
								mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
								name='mask_rcnn')	


			# Add multi-GPU support.  해당 안됨
			#if GPU_COUNT > 1:
			#	from mrcnn.parallel_model import ParallelModel
			#	model = ParallelModel(model, GPU_COUNT)
				
			return model

	def get_anchors(self, image_shape):
		"""
		image_shape : [1024, 1024, 3]
		
		Returns anchor pyramid for the given image size.
		Returns : [anchor_counts, (y1, x1, y2, x2)]
			y1, x1, y2, x2 has normalized
		"""
		backbone_shapes = utils.compute_backbone_shapes(image_shape, self.config.BACKBONE,
														self.config.FPN_PYAMID_STRIDES)
			
		# anchors 저장, input image의 shape이 동일하면 재사용
		if not hasattr(self, "_anchor_cache"): 	# _anchor_cache 변수가 없으면 선언
			self._anchor_cache = {}
		if not tuple(image_shape) in self._anchor_cache:	# anchor없으면 만들기.
			# Generate Anchors
			# anchor = [anchor_counts, (y1, x1, y2, x2)]
			anchor = utils.generate_pyramid_anchors(
					self.config.RPN_ANCHOR_SCALES,
					self.config.RPN_ANCHOR_RATIOS,
					backbone_shapes,
					self.config.BACKBONE_STRIDES,
					self.config.RPN_ANCHOR_STRIDE)
			
			# Keep a copy of the latest anchors in pixel coordinates becaus리
			# inspect_model notebooks에서 latest anchors in pixel coordinates가 사용되기 때문에
			# TODO : 뭔소리
			# 미리 copy해서 보관할것 ?
			self.anchors = anchor
			
			# Normalize coordinates
			# self._anchor_cache = {(1024, 1024, 3): [anchor_counts, (y1, x1, y2, x2)]}
			self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(anchor, image_shape[:2])
			
			return self._anchor_cache[tuple(image_shape)]


	def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
			  custom_callbacks=None):
		"""
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs.
		layers: Allows selecting wich layers to train.
				layers = "heads" : The RPN, classifier and mask heads of the network
				layers = "all" : All the layers
				layers = "3+": Train Resnet stage 3 and up
				layers = "4+": Train Resnet stage 4 and up
				layers = "5+": Train Resnet stage 5 and up
		custom_callbacks : Optional.
			Add custom callbacks to be called with the keras fit_generator method.
			Must be list of type keras.callbacks.
		"""

		assert self.mode == "training", "Create model in training mode."

		# Set the keys for which an expression targets a layer
		layer_regex = {
			# all layers but the backbone
			"heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",

			# From a specific Resnet stage and up
			"3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
			"4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
			"5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",

			# All layers
			"all": ".*",
		}
		if layers in layer_regex.keys():
			layers = layer_regex[layers]

		train_generator = data_generator(train_dataset, self.config, shuffle=True
										,batch_size=self.config.BATCH_SIZE)


