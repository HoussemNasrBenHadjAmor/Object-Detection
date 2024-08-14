import {
  getPredictionYolov9,
  getPredictionFRCNN,
  getPredictionYolov8,
} from "@/api";
import { useMutation } from "@tanstack/react-query";

const useYolov8 = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionYolov8({ image, threshold }),
  });
};

const useYolov9 = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionYolov9({ image, threshold }),
  });
};

const useYolov10 = () => {
  return useMutation({
    mutationFn: getPredictionYolov9,
  });
};

const useFrcnn = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionFRCNN({ image, threshold }),
  });
};

export { useYolov9, useYolov10, useYolov8, useFrcnn };
