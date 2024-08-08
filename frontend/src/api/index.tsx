export const getPredictionYolov9 = async ({
  image,
  threshold,
}: {
  image: File | null;
  threshold: number;
}) => {
  // Default threshold is 0.5
  if (!image) {
    throw new Error("No image provided");
  }

  const formData = new FormData();
  formData.append("file", image);

  const url = new URL("http://127.0.0.1:8000/detect/yolov9");
  url.searchParams.append("threshold", threshold.toString());

  const res = await fetch(url.toString(), {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Error uploading image");
  }

  const blob = await res.blob();
  const url_image = URL.createObjectURL(blob);

  return url_image;
};

export const getPredictionYolov8 = async ({
  image,
  threshold,
}: {
  image: File | null;
  threshold: number;
}) => {
  // Default threshold is 0.5
  if (!image) {
    throw new Error("No image provided");
  }

  const formData = new FormData();
  formData.append("file", image);

  const url = new URL("http://127.0.0.1:8000/detect/yolov8");
  url.searchParams.append("threshold", threshold.toString());

  const res = await fetch(url.toString(), {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Error uploading image");
  }

  const blob = await res.blob();
  const url_image = URL.createObjectURL(blob);

  return url_image;
};

export const getPredictionYolov10 = async (image: File | null) => {
  if (!image) {
    throw new Error("No image provided");
  }

  const formData = new FormData();
  formData.append("file", image);

  const res = await fetch("http://127.0.0.1:8000/detect/", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Error uploading image");
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  return url;
};

export const getPredictionFRCNN = async ({
  image,
  threshold,
}: {
  image: File | null;
  threshold: number;
}) => {
  // Default threshold is 0.5
  if (!image) {
    throw new Error("No image provided");
  }

  const formData = new FormData();
  formData.append("file", image);

  const url = new URL("http://127.0.0.1:8000/detect/frcnn");
  url.searchParams.append("threshold", threshold.toString());

  const res = await fetch(url.toString(), {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Error uploading image");
  }

  const blob = await res.blob();
  const url_image = URL.createObjectURL(blob);

  return url_image;
};
