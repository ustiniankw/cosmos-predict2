# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This directory vendors minimal code from the upstream Wan2.1 project:
#   https://github.com/Wan-Video/Wan2.1
#
# We vendor only what is needed to run the Wan2.1 video autoencoder inside Cosmos-Predict2.
#
# Upstream copyright:
#   Copyright 2024-2025 The Alibaba Wan Team Authors.

from .vae import WanVAE, WanVAE_  # noqa: F401

