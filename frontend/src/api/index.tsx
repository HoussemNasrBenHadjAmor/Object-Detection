export const getPredictionYolov9WithAttack = async ({
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

  const url = new URL("http://127.0.0.1:8000/detect/yolov9-with-attack");
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

export const getPredictionYolov9NoAttack = async ({
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

  const url = new URL("http://127.0.0.1:8000/detect/yolov9-no-attack");
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

export const getPredictionFRCNNWithAttack = async ({
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

  const url = new URL("http://127.0.0.1:8000/detect/frcnn-with-attack");
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

export const getPredictionFRCNNNoAttack = async ({
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

  const url = new URL("http://127.0.0.1:8000/detect/frcnn-no-attack");
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

export const getPredictionSSDWithAttack = async ({
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

  const url = new URL("http://127.0.0.1:8000/detect/ssd-with-attack");
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

export const getPredictionSSDNoAttack = async ({
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

  const url = new URL("http://127.0.0.1:8000/detect/ssd-no-attack");
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
