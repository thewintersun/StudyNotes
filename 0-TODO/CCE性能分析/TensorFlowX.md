## TensorFlow Extended (TFX) 

官方地址：https://www.tensorflow.org/tfx

TensorFlow Extended (TFX) 是一个端到端平台，用于部署生产型机器学习流水线。当您准备好将模型从研究状态切换到生产状态时，可以使用 TFX 创建和管理生产流水线。

### 工作原理

当您准备好训练多个模型，或准备将您的精彩模型投入使用并部署到生产环境中时，TFX 可以帮助您构建完整的机器学习流水线。

TFX 流水线是一系列实现机器学习流水线的组件，专门用于可扩展的高性能机器学习任务。这包括针对在线、原生移动和 JavaScript 目标建模、训练、运行推断和管理部署。要了解详情，请阅读我们的 [TFX 用户指南](https://www.tensorflow.org/tfx/guide)。

流水线组件使用 TFX 库构建而成，这些库也可以单独使用。下面概述了这些底层库。

**TensorFlow Data Validation**

[开始使用  ](https://www.tensorflow.org/tfx/guide/tfdv)

TensorFlow Data Validation (TFDV) 能够帮助开发者大规模地理解、验证和监控机器学习数据。Google 每天都使用 TFDV 分析和验证 PB 级的数据，并且在帮助 TFX 用户维护机器学习流水线正常运行方面，TFDV 一贯表现良好。

**TensorFlow Transform**

[开始使用  ](https://www.tensorflow.org/tfx/guide/tft)

在将机器学习应用于现实世界的数据集时，需要投入很多精力才能==将数据预处理为合适的格式==，其中包括在各种格式之间进行转换、对文本进行标记化和词干化、创建词汇表、执行归一化等各种数字操作。您可以使用 tf.Transform 完成所有这些操作。

**TensorFlow Model Analysis**

[开始使用  ](https://www.tensorflow.org/tfx/guide/tfma)

TensorFlow Model Analysis (TFMA) 让开发者能够计算和可视化模型的评估指标。在部署任何机器学习 (ML) 模型之前，机器学习开发者需要==评估模型的性==能，以确保其达到特定的质量阈值，并且能够==针对所有相关数据切片展示出与预期相符的行为==。例如，模型针对整个评估数据集的 AUC 可能是可接受的，但针对特定切片却表现不佳。TFMA 为开发者提供了==深入了解其模型性能的工具==。

**TensorFlow Serving**

[开始使用  ](https://www.tensorflow.org/tfx/guide/serving)

机器学习 (ML) 应用系统必须支持==模型版本控制==（用于更新模型并提供回滚选项）和多个模型（用于通过 A/B 测试进行实验），同时还要确保==并发模型==能够在硬件加速器（GPU 和 TPU）上==以较低的延迟实现较高的吞吐量==。TensorFlow Serving 在 Google 每秒能完成数千万次推断。



> 总结：为多个模型提供统一的调用接口，提供版本控制，批量推理等功能。



## Serving Models

### **Introduction**

TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data.

Detailed developer documentation on TensorFlow Serving is available:

- [Architecture Overview](https://www.tensorflow.org/tfx/serving/architecture)
- [Server API](https://www.tensorflow.org/tfx/serving/api_docs/cc/)
- [REST Client API](https://www.tensorflow.org/tfx/serving/api_rest)

### **Key Concepts**

To understand the architecture of TensorFlow Serving, you need to understand the following key concepts:

**Servables** are the central abstraction in TensorFlow Serving. Servables are the underlying objects that clients use to perform computation (for example, a lookup or inference).

The size and granularity of a Servable is flexible. A single Servable might include anything from a single shard of a lookup table to a single model to a tuple of inference models. Servables can be of any type and interface, enabling flexibility and future improvements such as:

- streaming results
- experimental APIs
- asynchronous modes of operation

Servables do not manage their own lifecycle.

Typical servables include the following:

- a TensorFlow SavedModelBundle (`tensorflow::Session`)
- a lookup table for embedding or vocabulary lookups

**Servable Versions**

TensorFlow Serving can handle one or more **versions** of a servable over the lifetime of a single server instance. This enables fresh algorithm configurations, weights, and other data to be loaded over time. Versions enable more than one version of a servable to be loaded concurrently, supporting gradual rollout and experimentation. At serving time, clients may request either the latest version or a specific version id, for a particular model.

**Servable Streams**

A **servable stream** is the sequence of versions of a servable, sorted by increasing version numbers.

Models

TensorFlow Serving represents a **model** as one or more servables. A machine-learned model may include one or more algorithms (including learned weights) and lookup or embedding tables.

You can represent a **composite model** as either of the following:

- multiple independent servables
- single composite servable

A servable may also correspond to a fraction of a model. For example, a large lookup table could be sharded across many TensorFlow Serving instances.

**Loaders** manage a servable's life cycle. The Loader API enables common infrastructure independent from specific learning algorithms, data or product use-cases involved. Specifically, Loaders standardize the APIs for loading and unloading a servable.

**Sources** are plugin modules that find and provide servables. Each Source provides zero or more servable streams. For each servable stream, a Source supplies one Loader instance for each version it makes available to be loaded. (A Source is actually chained together with zero or more SourceAdapters, and the last item in the chain emits the Loaders.)

TensorFlow Serving’s interface for Sources can discover servables from arbitrary storage systems. TensorFlow Serving includes common reference Source implementations. For example, Sources may access mechanisms such as RPC and can poll a file system.

Sources can maintain state that is shared across multiple servables or versions. This is useful for servables that use delta (diff) updates between versions.

**Aspired versions** represent the set of servable versions that should be loaded and ready. Sources communicate this set of servable versions for a single servable stream at a time. When a Source gives a new list of aspired versions to the Manager, it supercedes the previous list for that servable stream. The Manager unloads any previously loaded versions that no longer appear in the list.

See the [advanced tutorial](https://www.tensorflow.org/tfx/serving/serving_advanced) to see how version loading works in practice.

**Managers** handle the full lifecycle of Servables, including:

- loading Servables
- serving Servables
- unloading Servables

Managers listen to Sources and track all versions. The Manager tries to fulfill Sources' requests, but may refuse to load an aspired version if, say, required resources aren't available. Managers may also postpone an "unload". For example, a Manager may wait to unload until a newer version finishes loading, based on a policy to guarantee that at least one version is loaded at all times.

TensorFlow Serving Managers provide a simple, narrow interface -- `GetServableHandle()` -- for clients to access loaded servable instances.

**Core**

Using the standard TensorFlow Serving APis, *TensorFlow Serving Core* manages the following aspects of servables:

- lifecycle
- metrics

TensorFlow Serving Core treats servables and loaders as opaque objects.

Life of a Servable

![tf serving architecture diagram](https://www.tensorflow.org/tfx/serving/images/serving_architecture.svg?dcb_=0.3392394362381199)

Broadly speaking:

1. Sources create Loaders for Servable Versions.
2. Loaders are sent as Aspired Versions to the Manager, which loads and serves them to client requests.

In more detail:

1. A Source plugin creates a Loader for a specific version. The Loader contains whatever metadata it needs to load the Servable.
2. The Source uses a callback to notify the Manager of the Aspired Version.
3. The Manager applies the configured Version Policy to determine the next action to take, which could be to unload a previously loaded version or to load the new version.
4. If the Manager determines that it's safe, it gives the Loader the required resources and tells the Loader to load the new version.
5. Clients ask the Manager for the Servable, either specifying a version explicitly or just requesting the latest version. The Manager returns a handle for the Servable.

For example, say a Source represents a TensorFlow graph with frequently updated model weights. The weights are stored in a file on disk.

1. The Source detects a new version of the model weights. It creates a Loader that contains a pointer to the model data on disk.
2. The Source notifies the Dynamic Manager of the Aspired Version.
3. The Dynamic Manager applies the Version Policy and decides to load the new version.
4. The Dynamic Manager tells the Loader that there is enough memory. The Loader instantiates the TensorFlow graph with the new weights.
5. A client requests a handle to the latest version of the model, and the Dynamic Manager returns a handle to the new version of the Servable.

### **Extensibility**

**Version Policies** specify the sequence of version loading and unloading within a single servable stream.

TensorFlow Serving includes two policies that accommodate most known use- cases. These are the Availability Preserving Policy (avoid leaving zero versions loaded; typically load a new version before unloading an old one), and the Resource Preserving Policy (avoid having two versions loaded simultaneously, thus requiring double the resources; unload an old version before loading a new one). For simple usage of TensorFlow Serving where the serving availability of a model is important and the resource costs low, the Availability Preserving Policy will ensure that the new version is loaded and ready before unloading the old one. For sophisticated usage of TensorFlow Serving, for example managing versions across multiple server instances, the Resource Preserving Policy requires the least resources (no extra buffer for loading new versions).

New **Sources** could support new filesystems, cloud offerings and algorithm backends. TensorFlow Serving provides some common building blocks to make it easy & fast to create new sources. For example, TensorFlow Serving includes a utility to wrap polling behavior around a simple source. Sources are closely related to Loaders for specific algorithms and data hosting servables.

See the [Custom Source](https://www.tensorflow.org/tfx/serving/custom_source) document for more about how to create a custom Source.

**Loaders** are the extension point for adding algorithm and data backends. TensorFlow is one such algorithm backend. For example, you would implement a new Loader in order to load, provide access to, and unload an instance of a new type of servable machine learning model. We anticipate creating Loaders for lookup tables and additional algorithms.

See the [Custom Servable](https://www.tensorflow.org/tfx/serving/custom_servable) document to learn how to create a custom servable.

**Batching** of multiple requests into a single request can significantly reduce the cost of performing inference, especially in the presence of hardware accelerators such as GPUs. TensorFlow Serving includes a request batching widget that lets clients easily batch their type-specific inferences across requests into batch requests that algorithm systems can more efficiently process. See the [Batching Guide](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/batching/README.md) for more information.