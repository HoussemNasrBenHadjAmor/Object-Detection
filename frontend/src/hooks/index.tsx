import {
  getPredictionYolov9NoAttack,
  getPredictionYolov9WithAttack,
  getPredictionFRCNNWithAttack,
  getPredictionFRCNNNoAttack,
  getPredictionSSDWithAttack,
  getPredictionSSDNoAttack,
  getPredictionYolov8,
} from "@/api";
import { useMutation } from "@tanstack/react-query";

const useYolov9WithAttack = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionYolov9WithAttack({ image, threshold }),
  });
};

const useYolov9NoAttack = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionYolov9NoAttack({ image, threshold }),
  });
};

const useYolov8 = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionYolov8({ image, threshold }),
  });
};

const useFrcnnWithAttack = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionFRCNNWithAttack({ image, threshold }),
  });
};

const useFrcnnNoAttack = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionFRCNNNoAttack({ image, threshold }),
  });
};

const useSddWithAttack = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionSSDWithAttack({ image, threshold }),
  });
};

const useSddNoAttack = (image: File | null, threshold: number) => {
  return useMutation({
    mutationFn: () => getPredictionSSDNoAttack({ image, threshold }),
  });
};

export {
  useYolov9WithAttack,
  useYolov9NoAttack,
  useFrcnnWithAttack,
  useFrcnnNoAttack,
  useSddWithAttack,
  useSddNoAttack,
  useYolov8,
};
